#!/usr/bin/env bash
# Run Sleepwalker on your own GPU server (not Colab). Everything is written to ./sleepwalker_runs/<run>/
# (logs/ is the light part for analysis, weights/ the heavy part). Re-running the same command resumes.
#
# One GPU, everything in one go:
#   WM_PRESET=pro CUDA_VISIBLE_DEVICES=0 ./run_server.sh        (the key is read from .env)
#
# Two GPUs (recommended for pro):
#   1) dataset in two shards, in parallel (each process builds its own parts, no judge yet):
#        WM_PRESET=pro WM_SHARD=0/2 CUDA_VISIBLE_DEVICES=0 ./run_server.sh
#        WM_PRESET=pro WM_SHARD=1/2 CUDA_VISIBLE_DEVICES=1 ./run_server.sh
#      wait for both (python3 sleepwalker.py --status), then
#   2) pretraining on one GPU (judge, VM, world model, pi0, evaluations):
#        WM_PRESET=pro WM_STAGE=pretrain CUDA_VISIBLE_DEVICES=0 ./run_server.sh
#   3) experiment, one arm per GPU, in parallel:
#        WM_PRESET=pro WM_STAGE=experiment WM_ARMS=full      CUDA_VISIBLE_DEVICES=0 ./run_server.sh
#        WM_PRESET=pro WM_STAGE=experiment WM_ARMS=grpo_text CUDA_VISIBLE_DEVICES=1 ./run_server.sh
#
# Status without staying logged in:  python3 sleepwalker.py --status   (or read sleepwalker_runs/<run>/logs/status.json)
# Graceful stop at the next safe point: echo '{"stop": true}' > sleepwalker_runs/<run>/logs/control.json
set -euo pipefail
cd "$(dirname "$0")"
: "${WM_PRESET:=fast}"
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .env ]; then   # the key lives in .env (git-ignored): OPENROUTER_API_KEY=...
  set -a; . ./.env; set +a
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is not set: put OPENROUTER_API_KEY=... into .env next to this script (or export it)" >&2; exit 1
fi
if [ ! -d venv ]; then
  python3 -m venv venv
  ./venv/bin/pip install -q -U pip
  ./venv/bin/pip install -q torch transformers peft accelerate datasets
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tag="${WM_PRESET}${WM_STAGE:+_$WM_STAGE}${WM_SHARD:+_shard${WM_SHARD//\//of}}${WM_ARMS:+_$WM_ARMS}"
nohup ./venv/bin/python3 sleepwalker.py > "run_${tag}.out" 2>&1 &
echo "started, pid $!; follow: tail -f run_${tag}.out   or   python3 sleepwalker.py --status"
