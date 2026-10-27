#!/bin/bash
#SBATCH --job-name=act-scan
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/fembaptq/BioFoundation/act-scan-%j.log

# 激活量化的 PTQ 扫描（不训练，只用验证集）：检查重写的前向、选分位数、看各类位置的敏感度。
# 用法：sbatch scan_act_quant.sbatch [模式] [checkpoint] [名字] [权重位宽]
#   模式：full（默认，全部扫描）或 quick（检查、统一分位数的扫描、每次只量化一类的敏感度，几分钟）
#   checkpoint：一个微调好的浮点模型；不写或写 - 就用浮点参照（fp_ref）最近一次微调的最佳 checkpoint
#   名字：用在输出文件名里，默认 fp_ref
#   权重位宽：0（默认，权重保持浮点）、8 或 4
# 例：sbatch scan_act_quant.sbatch quick
#     sbatch scan_act_quant.sbatch full
#     sbatch scan_act_quant.sbatch full - fp_ref 4
set -euo pipefail

export DATA_PATH=/projects/fembaptq/TUAR_subject
export CHECKPOINT_DIR=/projects/fembaptq/qap_runs
cd /projects/fembaptq/BioFoundation

MODE=${1:-full}
CKPT=${2:--}
NAME=${3:-fp_ref}
WBITS=${4:-0}
if [ "$CKPT" = "-" ]; then
  DIR=$(ls -td /projects/fembaptq/qap_runs/finetune_fp_ref/checkpoints/finetune_fp_ref/*/ 2>/dev/null | head -n 1)
  CKPT=$(ls -t "${DIR}"epoch=*.ckpt 2>/dev/null | head -n 1)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then echo "找不到 checkpoint：$CKPT"; exit 1; fi
FLAG=""
if [ "$MODE" = "quick" ]; then FLAG="--quick"; fi
echo "模式：$MODE；checkpoint：$CKPT；权重位宽：$WBITS"

PY=/projects/fembaptq/femba-env/bin/python
$PY -m models.act_quant
$PY -u scan_act_quant.py --checkpoint "$CKPT" --name "$NAME" \
  --out-dir /projects/fembaptq/results_subject --weight-bits "$WBITS" $FLAG
