#!/usr/bin/env bash
# Run Sleepwalker on your own GPU server (not Colab). No environment variables needed: the OpenRouter key is read from
# ./.env (OPENROUTER_API_KEY=...), everything else is a command-line option passed through to sleepwalker.py.
# Everything is written to ./sleepwalker_runs/<run>/ (logs/ — light, for analysis; weights/ — heavy).
# Re-running the same command resumes.
#
# One GPU, everything in one go:
#   ./run_server.sh --gpu 0                 (all 6200 training tasks; --tasks N for a smaller run, e.g. --tasks 500)
#
# Two GPUs (recommended for pro):
#   1) dataset in two shards, in parallel (each process builds its own parts, no judge yet):
#        ./run_server.sh --shard 0/2 --gpu 0
#        ./run_server.sh --shard 1/2 --gpu 1
#      wait for both (./run_server.sh --status), then
#   2) pretraining on one GPU (judge, VM, world model, pi0, evaluations):
#        ./run_server.sh --stage pretrain --gpu 0
#   3) experiment, one arm per GPU, in parallel:
#        ./run_server.sh --stage experiment --arms full      --gpu 0
#        ./run_server.sh --stage experiment --arms grpo_text --gpu 1
#
# Status (no need to stay logged in):   ./run_server.sh --status
# Graceful stop at the next safe point: echo '{"stop": true}' > sleepwalker_runs/<run>/logs/control.json
# Local self-check without GPU/models/network: ./run_server.sh --dry   (runs in the foreground)
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .env ] && [[ " $* " != *" --status "* ]] && [[ " $* " != *" --dry "* ]]; then
  echo "no .env: create it next to this script with a line  OPENROUTER_API_KEY=...  (the judge's key)" >&2; exit 1
fi
if [ ! -d venv ]; then
  python3 -m venv venv
  ./venv/bin/pip install -q -U pip
  ./venv/bin/pip install -q torch transformers peft accelerate datasets
fi
case " $* " in
  *" --status "*|*" --dry "*) exec ./venv/bin/python3 sleepwalker.py "$@" ;;
esac
tag=$(echo "$*" | tr -c 'A-Za-z0-9._-' '_' | sed 's/^_*//; s/_*$//')
nohup ./venv/bin/python3 sleepwalker.py "$@" > "run_${tag:-default}.out" 2>&1 &
echo "started, pid $!; follow: tail -f run_${tag:-default}.out   or   ./run_server.sh --status"
