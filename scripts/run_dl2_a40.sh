#!/bin/bash
# a40 上 M5 DL v2（多通道时空）训练驱动：LSTM + TCN 依次，-u 无缓冲便于实时监控。
cd ~/pm25_dl || exit 1
PY=~/miniconda3/envs/biomatrix/bin/python
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=8
rm -f dl2_done.flag

echo "=== DL2 LSTM 开始 $(date) ==="
$PY -u scripts/train_dl2.py --model lstm --epochs 15 --max-train 800000 > train_dl2_lstm.log 2>&1
echo "LSTM 退出码=$? $(date)"

echo "=== DL2 TCN 开始 $(date) ==="
$PY -u scripts/train_dl2.py --model tcn --epochs 15 --max-train 800000 > train_dl2_tcn.log 2>&1
echo "TCN 退出码=$? $(date)"

echo "ALLDONE $(date)" > dl2_done.flag
