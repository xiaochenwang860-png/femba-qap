#!/bin/bash
#SBATCH --job-name=base-qat
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/fembaptq/BioFoundation/qap-finetune-%j.log

# 基线这条线上的 QAT：从作者的预训练权重出发，在 TUAR 上带权重量化微调。
# 除了量化之外，设置和 FP32 基线那次训练完全相同（10 轮，按验证损失保留最佳）。
# 用法：sbatch finetune_tuar_base_qat.sbatch <位宽> [smoke]
#   位宽：8 或 4；0 = 不量化（等同于重新训练一次 FP32 基线）
# 例：sbatch finetune_tuar_base_qat.sbatch 4
#     sbatch finetune_tuar_base_qat.sbatch 8
set -euo pipefail

BITS=${1:?需要位宽：0、8 或 4}
MODE=${2:-full}
if [ "$BITS" = "0" ]; then NAME=base_fp; else NAME=base_qat_w${BITS}; fi
WEIGHTS=/projects/fembaptq/FEMBA_tiny_for_author_finetune.safetensors
if [ ! -f "$WEIGHTS" ]; then echo "找不到作者的预训练权重：$WEIGHTS"; exit 1; fi

EXTRA=""
EPOCHS=10
if [ "$MODE" = "smoke" ]; then
  NAME=${NAME}_smoke
  EPOCHS=1
  EXTRA="+trainer.limit_train_batches=64 +trainer.limit_val_batches=2"
fi

export DATA_PATH=/projects/fembaptq/TUAR_subject
export CHECKPOINT_DIR=/projects/fembaptq/qap_runs/finetune_${NAME}
mkdir -p "$CHECKPOINT_DIR"
cd /projects/fembaptq/BioFoundation

echo "微调：名字=$NAME 位宽=$BITS 预训练=$WEIGHTS 模式=$MODE 输出=$CHECKPOINT_DIR"

/projects/fembaptq/femba-env/bin/python -u run_train.py \
  +experiment=FEMBA_finetune \
  task=finetune_base_qat_task \
  pretrained_safetensors_path=${WEIGHTS} \
  pretrained_checkpoint_path=null \
  +qap_weight_bits=${BITS} \
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
