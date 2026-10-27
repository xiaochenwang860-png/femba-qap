#!/bin/bash
#SBATCH --job-name=act-smoke
#SBATCH --partition=cluster02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/fembaptq/BioFoundation/act-scan-smoke-%j.log

# 带激活量化的训练的冒烟测试：每一步只跑几批，确认预训练、QAP 微调、QAT 微调、评估都能跑通。
# 用法：sbatch smoke_act_train.sbatch [分位数]      默认 95（W4A4 用的值）
# 用 W4A4（最容易出问题的配置）。结果没有意义，只看能不能跑通。
set -uo pipefail

APCT=${1:-95}
export DATA_PATH=/projects/fembaptq/TUAR_subject
export CHECKPOINT_DIR=/projects/fembaptq/qap_runs
cd /projects/fembaptq/BioFoundation
JOB=${SLURM_JOB_ID:-0}
PY=/projects/fembaptq/femba-env/bin/python
RUNS=/projects/fembaptq/qap_runs
FP=$RUNS/pretrain_fp_cosine_50ep/final.ckpt
PRE=$RUNS/pretrain_w4a4_cosine_1ep_smoke/final.ckpt
FAILED=""

step() {  # 名字 日志 命令...
  local name=$1 log=$2; shift 2
  if "$@" > "$log" 2>&1; then echo "[$(date '+%H:%M:%S')] 通过：$name"
  else echo "[$(date '+%H:%M:%S')] !!!!! 失败：$name（见 $log）"; FAILED="$FAILED $name"; grep -a "Error\|error:" "$log" | tail -n 5; fi
}
best() { ls -t "$RUNS/finetune_$1/checkpoints/finetune_$1"/*/epoch=*.ckpt 2>/dev/null | head -n 1; }

rm -f "$PRE"
step "预训练 W4A4" "qap-pretrain-${JOB}_smoke_w4a4.log" bash pretrain_tuar_qap.sbatch 4 1 cosine smoke 4 "$APCT"
grep -a "\[QAP\]" "qap-pretrain-${JOB}_smoke_w4a4.log"
step "QAP 微调 W4A4" "qap-finetune-${JOB}_smoke_qap_w4a4.log" bash finetune_tuar_qap.sbatch qap_w4a4 4 "$PRE" smoke 4 "$APCT"
grep -a "\[QAP\]" "qap-finetune-${JOB}_smoke_qap_w4a4.log"
step "QAT 微调 W4A4" "qap-finetune-${JOB}_smoke_qat_w4a4.log" bash finetune_tuar_qap.sbatch qat_w4a4 4 "$FP" smoke 4 "$APCT"
grep -a "\[QAP\]" "qap-finetune-${JOB}_smoke_qat_w4a4.log"

for NAME in qap_w4a4_smoke qat_w4a4_smoke; do
  CKPT=$(best "$NAME")
  if [ -z "$CKPT" ]; then echo "!!!!! 没有找到 $NAME 的 checkpoint"; FAILED="$FAILED 评估-$NAME"; continue; fi
  step "评估 $NAME" "act-smoke-eval-${JOB}_$NAME.txt" $PY -u eval_femba.py --checkpoint "$CKPT" --name "smoke_$NAME" --out-dir "$RUNS/smoke_results" --split val
  grep -a "量化方式\|auroc  " "act-smoke-eval-${JOB}_$NAME.txt"
done
FPREF=$(best fp_ref)
step "评估 PTQ W4A4" "act-smoke-eval-${JOB}_ptq.txt" $PY -u eval_femba.py --checkpoint "$FPREF" --name smoke_ptq_w4a4 --out-dir "$RUNS/smoke_results" --split val --ptq-bits 4 --ptq-act-bits 4 --ptq-act-percentile "$APCT"
grep -a "量化方式\|auroc  " "act-smoke-eval-${JOB}_ptq.txt"

if [ -z "$FAILED" ]; then echo "冒烟测试：全部通过"; else echo "冒烟测试：有失败的步骤 ->$FAILED"; exit 1; fi
