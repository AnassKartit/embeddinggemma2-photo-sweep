#!/bin/sh
# One answer model at a time, each under the memory watchdog (kills it if free memory < 15% or it passes 14 GB).
cd "$(dirname "$0")/.."
for repo in mlx-community/gemma-4-E4B-it-qat-4bit mlx-community/Qwen3-4B-Instruct-2507-4bit mlx-community/Qwen3.5-9B-MLX-4bit mlx-community/gemma-4-12B-it-qat-4bit; do
  echo "=== $repo  (free $(memory_pressure | tail -1 | grep -o '[0-9]*%'))"
  .venv/bin/python rag/answer.py --eval "$repo" 2>&1 | grep -v -i warn &
  sleep 2; PID=$(pgrep -f "answer.py --eval $repo" | head -1)
  [ -n "$PID" ] && ~/Projects/localfit/scripts/memwatch.sh "$PID" 14 15
  wait
done
echo ANSWERS_DONE
