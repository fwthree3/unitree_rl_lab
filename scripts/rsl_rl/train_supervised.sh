#!/usr/bin/env bash
# M1b.0/#252: crash/hang supervisor for long G1 Run training.
#
# Replaces the Claude-session cron watchdog used in M1.2 (it repeatedly failed to
# fire for hours/days during unattended periods). Runs as a plain shell loop:
#   - launches train.py (resuming from the newest checkpoint under the experiment
#     log dir, or from --load_run/--checkpoint on the very first launch),
#   - every POLL_S seconds checks the process is alive and its log is growing,
#   - kills the whole process group on exit/crash or a stale log (hang), logs the
#     event, and relaunches from the newest checkpoint,
#   - if the same checkpoint dies within FAST_S seconds twice in a row, falls back
#     one checkpoint earlier (the "poisoned checkpoint" pattern from M1.2),
#   - stops once a checkpoint >= TARGET_ITER exists.
#
# Usage (from unitree_rl_lab/):
#   TARGET_ITER=105000 scripts/rsl_rl/train_supervised.sh \
#       --load_run 2026-09-20_04-12-38 --checkpoint model_74000_logstd.pt [extra train.py args]
# The first --load_run/--checkpoint pair is used only for the first launch.
# Stop it with:  kill <supervisor pid>   (its trap kills the training process group too)
set -u

PY=${PY:-/home/jhan/workspace/kairn_rl/.venv-isaac/bin/python}
TASK=${TASK:-Unitree-G1-29dof-Velocity}
EXP_DIR=${EXP_DIR:-logs/rsl_rl/unitree_g1_29dof_velocity}
TARGET_ITER=${TARGET_ITER:?set TARGET_ITER}
POLL_S=${POLL_S:-60}
STALE_S=${STALE_S:-900}   # 15 min without log growth = hang
FAST_S=${FAST_S:-180}     # death within 3 min = "fast" crash
SUP_DIR=${SUP_DIR:-$EXP_DIR/supervisor}
mkdir -p "$SUP_DIR"
EVENTS="$SUP_DIR/events.log"

first_run=""; first_ckpt=""; extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --load_run) first_run="$2"; shift 2 ;;
    --checkpoint) first_ckpt="$2"; shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$EVENTS"; }

child_pgid=""
cleanup() {
  if [[ -n "$child_pgid" ]]; then kill -9 -- "-$child_pgid" 2>/dev/null; fi
  log "supervisor stopped"
  exit 0
}
trap cleanup INT TERM

# newest checkpoint across all run folders: prints "<run> <file> <iter>"
latest_ckpt() {
  ls -t "$EXP_DIR"/*/model_*.pt 2>/dev/null | grep -v logstd | head -1 | \
    awk -F/ '{f=$NF; r=$(NF-1); it=f; gsub(/model_|\.pt/,"",it); print r, f, it}'
}
# checkpoint one save earlier than the given run/file (same folder, else any folder)
previous_ckpt() {
  local it=$1 prev=$(( $1 - 100 ))
  local p; p=$(ls -t "$EXP_DIR"/*/model_${prev}.pt 2>/dev/null | head -1)
  [[ -n "$p" ]] && echo "$(basename "$(dirname "$p")") model_${prev}.pt ${prev}"
}

fast_count=0; last_ckpt_key=""; override=""
while true; do
  if pgrep -f "scripts/rsl_rl/train.py" >/dev/null; then
    log "another train.py is already running -- refusing to launch a second one; retry in ${POLL_S}s"
    sleep "$POLL_S"; continue
  fi

  if [[ -n "$override" ]]; then
    read -r run ckpt iter <<<"$override"; override=""
  elif [[ -n "$first_run" ]]; then
    run=$first_run; ckpt=$first_ckpt; iter=$(echo "$ckpt" | grep -oE '[0-9]+' | head -1)
    first_run=""
  else
    read -r run ckpt iter <<<"$(latest_ckpt)"
  fi
  if [[ -n "${iter:-}" && "$iter" -ge "$TARGET_ITER" ]]; then
    log "target reached: $run/$ckpt (>= $TARGET_ITER)"; exit 0
  fi

  out="$SUP_DIR/train_$(date +%Y%m%d_%H%M%S).log"
  log "launch resume=$run/$ckpt log=$out"
  setsid "$PY" -u scripts/rsl_rl/train.py --task "$TASK" --num_envs 4096 --headless \
      --resume --load_run "$run" --checkpoint "$ckpt" "${extra[@]}" >"$out" 2>&1 &
  child=$!; child_pgid=$child; start=$(date +%s)

  reason=""
  while true; do
    sleep "$POLL_S" & wait $!   # interruptible, so the TERM trap runs immediately
    read -r _ _ cur <<<"$(latest_ckpt)"
    if [[ -n "${cur:-}" && "$cur" -ge "$TARGET_ITER" ]]; then
      log "target reached at iter $cur -- stopping training"; kill -9 -- "-$child_pgid" 2>/dev/null
      child_pgid=""; exit 0
    fi
    if ! kill -0 "$child" 2>/dev/null; then reason="exited"; break; fi
    if grep -qE "device-side assert|AcceleratorError|CUDA error" "$out"; then reason="cuda-assert"; break; fi
    age=$(( $(date +%s) - $(stat -c %Y "$out") ))
    if (( age > STALE_S )); then reason="hang(log stale ${age}s)"; break; fi
  done

  kill -9 -- "-$child_pgid" 2>/dev/null; child_pgid=""
  sleep 10
  alive=$(( $(date +%s) - start ))
  read -r lrun lckpt liter <<<"$(latest_ckpt)"
  log "event=$reason alive=${alive}s resumed_from=$run/$ckpt latest=$lrun/$lckpt"

  key="$iter"   # iteration, not path: every resume writes a new run folder with a copy at the same iteration
  if (( alive < FAST_S )) && [[ "$key" == "$last_ckpt_key" ]]; then
    fast_count=$((fast_count + 1))
  elif (( alive < FAST_S )); then
    fast_count=1
  else
    fast_count=0
  fi
  last_ckpt_key=$key
  if (( fast_count >= 2 )); then
    prev=$(previous_ckpt "$iter")
    if [[ -n "$prev" ]]; then
      log "two fast deaths on $key -- falling back to $prev"; override=$prev; fast_count=0
    fi
  fi
done
