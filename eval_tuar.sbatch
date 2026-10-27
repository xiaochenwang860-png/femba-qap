#!/bin/bash
#SBATCH --job-name=eval-tuar
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/fembaptq/BioFoundation/eval-tuar-%j.log

# 评估一个微调得到的 checkpoint，结果写到 /projects/fembaptq/results_subject。
# 用法：sbatch eval_tuar.sbatch <checkpoint> <名字> <val|test|both> [PTQ位宽] [max|mse]
#   PTQ位宽：8 或 4，把浮点 checkpoint 的权重量化到这个位宽后再评估（PTQ）；不写就按 checkpoint 原样评估
#   max|mse：PTQ 的步长怎么定，默认 max
# 例：sbatch eval_tuar.sbatch /path/to/fp_ref.ckpt fp_ref val
#     sbatch eval_tuar.sbatch /path/to/fp_ref.ckpt ptq_w8 val 8
#     sbatch eval_tuar.sbatch /path/to/fp_ref.ckpt ptq_w4_max val 4
#     sbatch eval_tuar.sbatch /path/to/fp_ref.ckpt ptq_w4_mse val 4 mse
#     sbatch eval_tuar.sbatch /path/to/qap_w4.ckpt qap_w4 val
set -euo pipefail

CKPT=${1:?需要 checkpoint 路径}
NAME=${2:?需要名字}
SPLIT=${3:?需要 val、test 或 both}
PTQ=${4:-}
INIT=${5:-max}

FLAG=""
if [ -n "$PTQ" ]; then FLAG="--ptq-bits $PTQ --ptq-init $INIT"; fi
if [ "$SPLIT" = "both" ]; then SPLITS="val test"; else SPLITS="$SPLIT"; fi

export DATA_PATH=/projects/fembaptq/TUAR_subject
export CHECKPOINT_DIR=/projects/fembaptq/qap_runs
cd /projects/fembaptq/BioFoundation

for S in $SPLITS; do
  /projects/fembaptq/femba-env/bin/python -u eval_femba.py \
    --checkpoint "$CKPT" --name "$NAME" --out-dir /projects/fembaptq/results_subject --split "$S" $FLAG
done
