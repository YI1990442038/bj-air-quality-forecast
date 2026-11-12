#!/bin/bash
# a40 上的 M5 训练驱动：LSTM + TCN 依次训练，日志落盘，完成写标记文件。
cd ~/pm25_dl || exit 1
PY=~/miniconda3/envs/biomatrix/bin/python
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=8
rm -f train_done.flag

echo "=== LSTM 开始 $(date) ==="
$PY scripts/train_dl.py --model lstm --epochs 15 --max-train 800000 > train_lstm.log 2>&1
echo "LSTM 退出码=$? $(date)"

echo "=== TCN 开始 $(date) ==="
$PY scripts/train_dl.py --model tcn --epochs 15 --max-train 800000 > train_tcn.log 2>&1
echo "TCN 退出码=$? $(date)"

echo "ALLDONE $(date)" > train_done.flag
