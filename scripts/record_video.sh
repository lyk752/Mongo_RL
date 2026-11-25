#!/usr/bin/env bash
#
# Record a short clip from a trained checkpoint with the Mongo-Flat task.
#
# This is the recommended way to get training videos: keep the long training run
# free of `--video` (it is much faster) and record clips on demand from the
# checkpoints that the training run keeps saving.
#
# Usage:
#   scripts/record_video.sh                          # newest checkpoint under logs/
#   scripts/record_video.sh path/to/model_5000.pt    # a specific checkpoint
#   NUM_ENVS=4 STEPS=600 scripts/record_video.sh     # overrides
#
# Environment overrides:
#   ISAACLAB_SH   path to isaaclab.sh        (default: /home/inin/workspace/IsaacLab/isaaclab.sh)
#   TASK          task id                    (default: Mongo-Flat-v0)
#   NUM_ENVS      parallel envs to record    (default: 1)
#   STEPS         how many policy steps      (default: 400, i.e. 8 s at 50 Hz)
#   VIDEO_LEN     frames captured            (default: STEPS)
#   VIEWER_ROBOT  1 = close-up on robot 0, 0 = task default world camera
#                 (default: 1, i.e. a close-up clip of the trained policy)
#   EXTRA_ARGS    extra play.py arguments
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ISAACLAB_SH="${ISAACLAB_SH:-/home/inin/workspace/IsaacLab/isaaclab.sh}"
TASK="${TASK:-Mongo-Flat-v0}"
NUM_ENVS="${NUM_ENVS:-1}"
STEPS="${STEPS:-400}"
VIDEO_LEN="${VIDEO_LEN:-$STEPS}"
VIEWER_ROBOT="${VIEWER_ROBOT:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

VIEWER_ARGS=()
if [[ "$VIEWER_ROBOT" == "1" ]]; then
    VIEWER_ARGS+=(--viewer_robot)
fi

CKPT="${1:-}"
if [[ -z "$CKPT" ]]; then
    CKPT="$(ls -t logs/rsl_rl/mongo_flat_direct/*/model_*.pt 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "[record_video] no checkpoint found." >&2
    echo "[record_video] usage: scripts/record_video.sh [path/to/model_XXXX.pt]" >&2
    exit 1
fi

echo "[record_video] checkpoint : $CKPT"
echo "[record_video] task       : $TASK   (num_envs=$NUM_ENVS, steps=$STEPS)"
echo "[record_video] camera     : $([[ "$VIEWER_ROBOT" == "1" ]] && echo 'follow robot 0' || echo 'world / multi-robot')"
echo "[record_video] note       : the first run in a container without a display"
echo "[record_video]              spends several minutes warming up the RTX"
echo "[record_video]              renderer before anything is recorded."

# shellcheck disable=SC2086
"$ISAACLAB_SH" -p scripts/rsl_rl/play.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --checkpoint "$CKPT" \
    --headless \
    --video \
    --video_length "$VIDEO_LEN" \
    --max_steps "$STEPS" \
    "${VIEWER_ARGS[@]}" \
    $EXTRA_ARGS

echo "[record_video] done. videos:"
find "$(dirname "$CKPT")/videos" -name '*.mp4' -printf '  %p\n' 2>/dev/null || true
