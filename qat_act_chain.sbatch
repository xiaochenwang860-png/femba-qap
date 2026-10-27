#!/bin/bash
#SBATCH --job-name=qat-act
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/fembaptq/BioFoundation/qap-chain-%j.log

# 带激活量化的 QAT：从已有的浮点预训练权重出发，依次微调 W8A8、W4A4、W4A8。
# 用法：sbatch qat_act_chain.sbatch
# 激活范围校准用的分位数按激活位宽取（验证集上用 PTQ 选出）：8 bit 用 99.9，4 bit 用 95。
set -uo pipefail

cd /projects/fembaptq/BioFoundation
JOB=${SLURM_JOB_ID:-0}
PRE=/projects/fembaptq/qap_runs/pretrain_fp_cosine_50ep/final.ckpt
if [ ! -f "$PRE" ]; then echo "找不到浮点预训练的权重：$PRE"; exit 1; fi

FAILED=0
for CFG in "8 8" "4 4" "4 8"; do
  set -- $CFG
  if [ "$2" = "8" ]; then APCT=99.9; else APCT=95; fi
  NAME=qat_w$1a$2
  echo "[$(date '+%H:%M:%S')] 开始微调：$NAME（权重 $1，激活 $2，分位数 $APCT）"
  if bash finetune_tuar_qap.sbatch "$NAME" "$1" "$PRE" full "$2" "$APCT" > "qap-finetune-${JOB}_${NAME}.log" 2>&1; then
    echo "[$(date '+%H:%M:%S')] 微调完成：$NAME"
  else
    echo "[$(date '+%H:%M:%S')] !!!!! 微调失败：$NAME，见 qap-finetune-${JOB}_${NAME}.log"
    FAILED=1
  fi
done
echo "[$(date '+%H:%M:%S')] 这一串任务结束"
exit $FAILED
