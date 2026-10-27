#!/bin/bash
#SBATCH --job-name=qap-chain
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/projects/fembaptq/BioFoundation/qap-chain-%j.log

# 在一个任务里连着跑：一次预训练（50 轮）+ 从它出发的微调。
# 用法：sbatch chain_qap.sbatch <权重位宽> [激活位宽] [分位数]
#   0            浮点预训练 -> 浮点参照、QAT W4、QAT W8 三个微调
#   8 或 4       权重量化预训练 -> QAP 微调（qap_w8 / qap_w4）
#   8 8 99.99    权重 8 bit、激活 8 bit 预训练 -> QAP 微调（qap_w8a8）；W4A8、W4A4 同理
# 每一步的输出单独写一个日志：qap-pretrain-<任务号>_<名字>.log、qap-finetune-<任务号>_<名字>.log
set -uo pipefail

BITS=${1:?需要权重位宽：0、8 或 4}
ABITS=${2:-0}
APCT=${3:-99.999}
EPOCHS=50
cd /projects/fembaptq/BioFoundation
JOB=${SLURM_JOB_ID:-0}
if [ "$BITS" = "0" ]; then NAME=fp; else NAME=w${BITS}; fi
if [ "$ABITS" != "0" ]; then
  if [ "$BITS" = "0" ]; then NAME=w32a${ABITS}; else NAME=w${BITS}a${ABITS}; fi
fi
PRE=/projects/fembaptq/qap_runs/pretrain_${NAME}_cosine_${EPOCHS}ep/final.ckpt

echo "[$(date '+%H:%M:%S')] 开始预训练：权重位宽=$BITS 激活位宽=$ABITS 分位数=$APCT"
rm -f "$PRE"   # 确保后面用到的一定是这次训练保存的
bash pretrain_tuar_qap.sbatch "$BITS" "$EPOCHS" cosine full "$ABITS" "$APCT" > "qap-pretrain-${JOB}_${NAME}.log" 2>&1
if [ $? -ne 0 ] || [ ! -f "$PRE" ]; then
  echo "[$(date '+%H:%M:%S')] 预训练失败，后面的微调不做。见 qap-pretrain-${JOB}_${NAME}.log"
  exit 1
fi
echo "[$(date '+%H:%M:%S')] 预训练完成：$PRE"

finetune() {  # 名字 权重位宽 激活位宽
  echo "[$(date '+%H:%M:%S')] 开始微调：$1（权重 $2，激活 $3）"
  if bash finetune_tuar_qap.sbatch "$1" "$2" "$PRE" full "$3" "$APCT" > "qap-finetune-${JOB}_$1.log" 2>&1; then
    echo "[$(date '+%H:%M:%S')] 微调完成：$1"
  else
    echo "[$(date '+%H:%M:%S')] !!!!! 微调失败：$1，见 qap-finetune-${JOB}_$1.log"
    FAILED=1
  fi
}

FAILED=0
if [ "$BITS" = "0" ] && [ "$ABITS" = "0" ]; then
  finetune fp_ref 0 0
  finetune qat_w4 4 0
  finetune qat_w8 8 0
else
  finetune "qap_${NAME}" "$BITS" "$ABITS"
fi
echo "[$(date '+%H:%M:%S')] 这一串任务结束"
exit $FAILED
