#!/bin/bash
#SBATCH --job-name=qap-finetune
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/fembaptq/BioFoundation/qap-finetune-%j.log

# 在 TUAR 上微调，设置和 FP32 基线相同（10 轮，按验证损失保留最佳）。
# 用法：sbatch finetune_tuar_qap.sbatch <名字> <位宽> <预训练 checkpoint 或 none> [full|smoke] [激活位宽] [分位数]
#   激活位宽：0（默认，激活浮点）、8 或 4；QAP 组要和预训练的相同
#   名字：这次运行的名字，用在输出目录里
#   位宽：0 = 浮点微调，8 或 4 = 带 8 bit / 4 bit 权重量化微调（QAP 组要和预训练的位宽相同）
#   none：不加载任何预训练权重，从随机初始化直接微调（位宽只能是 0）
# 例：sbatch finetune_tuar_qap.sbatch fp_ref 0 /path/to/pretrain_fp.ckpt     浮点参照（也用于 PTQ）
#     sbatch finetune_tuar_qap.sbatch qat_w8 8 /path/to/pretrain_fp.ckpt     QAT
#     sbatch finetune_tuar_qap.sbatch qap_w8 8 /path/to/pretrain_w8.ckpt     QAP
#     sbatch finetune_tuar_qap.sbatch qat_w4 4 /path/to/pretrain_fp.ckpt     QAT，4 bit
#     sbatch finetune_tuar_qap.sbatch qap_w4 4 /path/to/pretrain_w4.ckpt     QAP，4 bit
#     sbatch finetune_tuar_qap.sbatch scratch 0 none                         不预训练
set -euo pipefail

NAME=${1:?需要名字}
BITS=${2:?需要位宽：0、8 或 4}
CKPT=${3:?需要预训练 checkpoint 的路径，或 none}
MODE=${4:-full}
ABITS=${5:-0}       # 激活位宽：0（浮点）、8 或 4
APCT=${6:-99.999}   # 激活范围校准用的分位数

EXTRA=""
EPOCHS=10
if [ "$MODE" = "smoke" ]; then
  NAME=${NAME}_smoke
  EPOCHS=1
  EXTRA="+trainer.limit_train_batches=64 +trainer.limit_val_batches=2"
fi

if [ "$CKPT" = "none" ]; then
  if [ "$BITS" != "0" ]; then echo "不预训练的参照只支持位宽 0"; exit 1; fi
  TASK_ARGS="task=finetune_task pretrained_safetensors_path=null pretrained_checkpoint_path=null"
else
  if [ ! -f "$CKPT" ]; then echo "找不到 checkpoint：$CKPT"; exit 1; fi
  TASK_ARGS="task=finetune_qap_task pretrained_safetensors_path=null pretrained_checkpoint_path=$CKPT +qap_weight_bits=$BITS +qap_act_bits=$ABITS +qap_act_percentile=$APCT"
fi

export DATA_PATH=/projects/fembaptq/TUAR_subject
export CHECKPOINT_DIR=/projects/fembaptq/qap_runs/finetune_${NAME}
mkdir -p "$CHECKPOINT_DIR"
cd /projects/fembaptq/BioFoundation

echo "微调：名字=$NAME 位宽=$BITS 激活位宽=$ABITS 分位数=$APCT 预训练=$CKPT 模式=$MODE 输出=$CHECKPOINT_DIR"

/projects/fembaptq/femba-env/bin/python -u run_train.py \
  +experiment=FEMBA_finetune \
  ${TASK_ARGS} \
  tag=finetune_${NAME} \
  model.embed_dim=35 \
  model.num_classes=2 \
  model.classification_type=bc \
  data_module.train.hdf5_file=/projects/fembaptq/TUAR_subject/train.h5 \
  data_module.val.hdf5_file=/projects/fembaptq/TUAR_subject/val.h5 \
  data_module.test.hdf5_file=/projects/fembaptq/TUAR_subject/test.h5 \
  gpus=1 \
  trainer.devices=1 \
  trainer.strategy=auto \
  batch_size=32 \
  +trainer.accumulate_grad_batches=32 \
  num_workers=0 \
  trainer.max_epochs=${EPOCHS} \
  final_validate=False \
  final_test=False \
  io.base_output_path=${CHECKPOINT_DIR} \
  resume=False \
  ${EXTRA}

echo "微调结束。checkpoint："
find "$CHECKPOINT_DIR/checkpoints" -name "*.ckpt" | sort
