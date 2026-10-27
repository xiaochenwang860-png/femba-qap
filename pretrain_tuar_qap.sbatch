#!/bin/bash
#SBATCH --job-name=qap-pretrain
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/fembaptq/BioFoundation/qap-pretrain-%j.log

# 在 TUAR 训练集上做掩码重建预训练，从随机初始化开始。
# 用法：sbatch pretrain_tuar_qap.sbatch <位宽> <轮数> <学习率调度> [full|smoke] [激活位宽] [分位数]
#   激活位宽：0（默认，激活浮点）、8 或 4；分位数：激活范围校准用，默认 99.999
#   位宽：0 = 浮点（对照组），8 或 4 = 编码器权重 8 bit / 4 bit 量化（QAP 组）
#   学习率调度：cosine = 预热 2 轮后余弦衰减（正式实验），const = 全程固定（试跑）
#   smoke：每轮只跑 6 个训练批和 2 个验证批，用来确认代码能运行
# 例：sbatch pretrain_tuar_qap.sbatch 0 4 cosine smoke
#     sbatch pretrain_tuar_qap.sbatch 0 20 const
#     sbatch pretrain_tuar_qap.sbatch 0 50 cosine
#     sbatch pretrain_tuar_qap.sbatch 8 50 cosine
set -euo pipefail

BITS=${1:?需要位宽：0、8 或 4}
EPOCHS=${2:?需要轮数}
SCHED=${3:?需要学习率调度：cosine 或 const}
MODE=${4:-full}
ABITS=${5:-0}       # 激活位宽：0（浮点）、8 或 4
APCT=${6:-99.999}   # 激活范围校准用的分位数
if [ "$SCHED" != "cosine" ] && [ "$SCHED" != "const" ]; then
  echo "学习率调度只能是 cosine 或 const，收到 $SCHED"; exit 1
fi
if [ "$BITS" = "0" ]; then NAME=fp; else NAME=w${BITS}; fi
if [ "$ABITS" != "0" ]; then
  if [ "$BITS" = "0" ]; then NAME=w32a${ABITS}; else NAME=w${BITS}a${ABITS}; fi
fi
NAME=${NAME}_${SCHED}_${EPOCHS}ep

EXTRA=""
if [ "$MODE" = "smoke" ]; then
  NAME=${NAME}_smoke
  EXTRA="+trainer.limit_train_batches=6 +trainer.limit_val_batches=2 trainer.log_every_n_steps=1"
fi

export DATA_PATH=/projects/fembaptq/TUAR_subject
export CHECKPOINT_DIR=/projects/fembaptq/qap_runs/pretrain_${NAME}
mkdir -p "$CHECKPOINT_DIR"
cd /projects/fembaptq/BioFoundation

echo "预训练：位宽=$BITS 激活位宽=$ABITS 分位数=$APCT 轮数=$EPOCHS 学习率调度=$SCHED 模式=$MODE 输出=$CHECKPOINT_DIR"

/projects/fembaptq/femba-env/bin/python -u run_train.py \
  +experiment=FEMBA_pretrain \
  task=pretrain_qap_task \
  data_module=pretrain_tuar_subject \
  tag=pretrain_${NAME} \
  model.embed_dim=35 \
  model.num_blocks=2 \
  gpus=1 \
  trainer.devices=1 \
  trainer.strategy=auto \
  batch_size=256 \
  num_workers=0 \
  trainer.max_epochs=${EPOCHS} \
  +qap_weight_bits=${BITS} \
  +qap_act_bits=${ABITS} \
  +qap_act_percentile=${APCT} \
  +qap_lr_schedule=${SCHED} \
  +qap_warmup_epochs=2 \
  final_validate=False \
  final_test=False \
  io.base_output_path=${CHECKPOINT_DIR} \
  resume=False \
  ${EXTRA}

echo "预训练结束。微调请用最后一轮的权重："
ls -l "$CHECKPOINT_DIR/final.ckpt"
