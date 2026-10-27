#!/bin/bash
#SBATCH --job-name=eval-all
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/projects/fembaptq/BioFoundation/eval-all-%j.log

# 一次评估 QAP 这条线上所有已经微调完的模型，结果写到 /projects/fembaptq/results_subject。
# 用法：sbatch eval_all_qap.sbatch [val|test|both]     默认 val
# 激活 PTQ 的分位数按激活位宽取（验证集上选出）：8 bit 用 99.9，4 bit 用 95。
# 对每个微调目录，自动取最近一次运行里验证损失最低的那个 checkpoint。还没微调完的组会跳过。
# 浮点参照（fp_ref）另外做 PTQ：权重 8 bit、4 bit（max）、4 bit（mse）；以及权重加激活的 W8A8、W4A8、W4A4。
set -uo pipefail

SPLIT=${1:-val}
if [ "$SPLIT" = "both" ]; then SPLITS="val test"; else SPLITS="$SPLIT"; fi

export DATA_PATH=/projects/fembaptq/TUAR_subject
export CHECKPOINT_DIR=/projects/fembaptq/qap_runs
cd /projects/fembaptq/BioFoundation
PY=/projects/fembaptq/femba-env/bin/python
OUT=/projects/fembaptq/results_subject
RUNS=/projects/fembaptq/qap_runs

best_ckpt() {  # 最近一次运行的目录里，除 last.ckpt 之外的那个 checkpoint
  local dir
  dir=$(ls -td "$RUNS/finetune_$1/checkpoints/finetune_$1"/*/ 2>/dev/null | head -n 1)
  [ -n "$dir" ] && ls -t "$dir"epoch=*.ckpt 2>/dev/null | head -n 1
}

run_eval() {  # 名字 checkpoint [其余参数]
  local name=$1 ckpt=$2; shift 2
  for S in $SPLITS; do
    echo "===== $name / $S ====="
    $PY -u eval_femba.py --checkpoint "$ckpt" --name "$name" --out-dir "$OUT" --split "$S" "$@" \
      || echo "!!!!! $name / $S 评估失败"
  done
}

for NAME in fp_ref qat_w8 qat_w4 qap_w8 qap_w4 scratch \
            qat_w8a8 qat_w4a8 qat_w4a4 qap_w8a8 qap_w4a8 qap_w4a4 base_qat_w8 base_qat_w4; do
  CKPT=$(best_ckpt "$NAME")
  if [ -z "$CKPT" ]; then echo "----- 跳过 $NAME：还没有 checkpoint"; continue; fi
  echo "----- $NAME 用的 checkpoint：$CKPT"
  run_eval "qapline_$NAME" "$CKPT"
  if [ "$NAME" = "fp_ref" ]; then
    run_eval qapline_ptq_w8 "$CKPT" --ptq-bits 8
    run_eval qapline_ptq_w4_max "$CKPT" --ptq-bits 4 --ptq-init max
    run_eval qapline_ptq_w4_mse "$CKPT" --ptq-bits 4 --ptq-init mse
    run_eval qapline_ptq_w8a8 "$CKPT" --ptq-bits 8 --ptq-act-bits 8 --ptq-act-percentile 99.9
    run_eval qapline_ptq_w4a8 "$CKPT" --ptq-bits 4 --ptq-act-bits 8 --ptq-act-percentile 99.9
    run_eval qapline_ptq_w4a4 "$CKPT" --ptq-bits 4 --ptq-act-bits 4 --ptq-act-percentile 95
  fi
done
echo "全部评估结束"
