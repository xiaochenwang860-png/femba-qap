"""把服务器上的训练、评估结果汇总成一份文档，省得逐个看日志。

读取：
  BioFoundation 目录下的 qap-pretrain-*.log、qap-finetune-*.log、eval-*.log
  /projects/fembaptq/results_subject 下的 *_metrics.json（评估指标）
写出（默认到 /projects/fembaptq/report/）：
  report.md              汇总文档：当前队列、评估指标表、每个任务的设置和每轮曲线
  curves/<任务>.csv       每个训练任务的逐轮数值

只用 Python 标准库，在登录节点上直接运行即可，不需要 GPU：
  python make_report.py
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

JOB_ID = re.compile(r"-(\d+)(?:_\w+)?\.log$")
NUMBER = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf)"
SKIP_KEYS = {"v_num", "train_loss_step"}


def run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def parse_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n") if line.strip()]
    job_id = JOB_ID.search(str(path))
    info = {"file": Path(path).name, "job_id": job_id.group(1) if job_id else "",
            "header": "", "notes": [], "epochs": {}, "best": "", "best_score": "",
            "error": [], "ended": False}
    for i, line in enumerate(lines):
        if not info["header"] and re.match(r"(预训练|微调)：", line):
            info["header"] = line
        if line.startswith("[QAP]"):
            if line not in info["notes"]:
                info["notes"].append(line)
        match = re.match(r"\s*Epoch (\d+):\s+100%", line)
        if match:  # 同一轮会出现多行，最后一行带有这一轮结束后的验证结果
            values = dict(re.findall(rf"(\w+)=({NUMBER})", line))
            if values:
                info["epochs"][int(match.group(1))] = values
        if line.startswith("Best checkpoint path:"):
            info["best"] = line.split(":", 1)[1].strip()
        if line.startswith("Best model score:"):
            info["best_score"] = line.split(":", 1)[1].strip()
        if "Traceback (most recent call last)" in line:
            info["error"] = lines[i:][-12:]
        if "CANCELLED" in line or "DUE TO TIME LIMIT" in line:
            info["error"] = [line]
        if re.match(r"(预训练|微调)结束", line):
            info["ended"] = True
    output = re.search(r"输出=(\S+)", info["header"])
    info["checkpoints"] = (sorted(glob.glob(os.path.join(output.group(1), "*.ckpt")))
                           + sorted(glob.glob(os.path.join(output.group(1), "checkpoints", "**", "*.ckpt"),
                                              recursive=True))) if output else []
    return info


def job_state(job_id):
    if not job_id:
        return ""
    out = run(["sacct", "-j", job_id, "-X", "-n", "-o", "State,Elapsed"])
    return " ".join(out.split()) if out else ""


def fmt(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "" if value is None else str(value)


def metrics_table(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_metrics.json"))):
        try:
            m = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        m["_mtime"] = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).strftime("%m-%d %H:%M")
        rows.append(m)
    if not rows:
        return ["（还没有评估结果）"]
    rows.sort(key=lambda m: (m.get("split", ""), m.get("name", "")))
    keys = ["auroc", "aupr", "accuracy", "balanced_accuracy", "f1", "ece", "loss", "auroc_author_style"]
    head = ["名字", "数据", "量化方式", "量化张量数"] + keys + ["AUROC 95% 区间", "checkpoint 轮", "生成时间(UTC)"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in rows:
        ci = m.get("auroc_ci95")
        out.append("| " + " | ".join(
            [str(m.get("name", "")), str(m.get("split", "")), str(m.get("quant", "")),
             str(m.get("n_quantized_weight_tensors", ""))]
            + [fmt(m.get(k)) for k in keys]
            + [f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "",
               str(int(m.get("checkpoint_epoch", -2)) + 1), m["_mtime"]]) + " |")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log-dir", default="/projects/fembaptq/BioFoundation")
    parser.add_argument("--results-dir", default="/projects/fembaptq/results_subject")
    parser.add_argument("--out-dir", default="/projects/fembaptq/report")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "curves").mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc = [f"# FEMBA 量化实验结果汇总", "", f"生成时间：{now}（新加坡时间 = UTC + 8 小时）", ""]

    doc += ["## 当前队列", "", "```"]
    queue = run(["squeue", "-u", os.environ.get("USER", ""), "-o", "%.8i %.13j %.3t %.10M %.18b %.20S %R"])
    doc += [queue or "（没有取到队列信息）", "```", ""]

    doc += ["## 评估指标", ""] + metrics_table(args.results_dir) + [""]

    doc += ["## 训练任务", ""]
    logs = sorted(glob.glob(os.path.join(args.log_dir, "qap-pretrain-*.log"))
                  + glob.glob(os.path.join(args.log_dir, "qap-finetune-*.log")),
                  key=lambda p: (int(JOB_ID.search(p).group(1)), p))
    overview = ["| 任务号 | 设置 | 状态 | 完成轮数 | 最后一轮验证损失 | 最低验证损失 |", "|---|---|---|---|---|---|"]
    details = []
    for path in logs:
        info = parse_log(path)
        epochs = info["epochs"]
        if "_" in Path(path).stem.split("-")[-1]:  # 串行任务里的一步：按这一步自己的日志判断
            state = "已结束" if info["ended"] else "进行中或中断"
        else:
            state = job_state(info["job_id"]) or ("已结束" if info["ended"] else "未结束或未知")
        if info["error"] and not info["ended"]:
            state += "；有报错或被取消"
        val = [(e, float(v["val_loss"])) for e, v in sorted(epochs.items()) if "val_loss" in v]
        last = f"{val[-1][1]:.4f}" if val else ""
        best = f"{min(val, key=lambda x: x[1])[1]:.4f}（第 {min(val, key=lambda x: x[1])[0] + 1} 轮）" if val else ""
        overview.append(f"| {info['job_id']} | {info['header']} | {state} | {len(epochs)} | {last} | {best} |")

        details += [f"### 任务 {info['job_id']}", "", info["header"] or info["file"], ""]
        details += [f"- {note}" for note in info["notes"]]
        details += [f"- 状态：{state}"]
        if info["best"]:
            details += [f"- 验证损失最低的 checkpoint：`{info['best']}`（{info['best_score']}）"]
        for ckpt in info["checkpoints"]:
            details += [f"- checkpoint 文件：`{ckpt}`"]
        if info["error"]:
            details += ["", "报错或中断信息：", "", "```"] + info["error"] + ["```"]
        if epochs:
            keys = [k for k in dict.fromkeys(k for v in epochs.values() for k in v) if k not in SKIP_KEYS]
            details += ["", "| 轮 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
            for e in sorted(epochs):
                details.append(f"| {e + 1} | " + " | ".join(epochs[e].get(k, "") for k in keys) + " |")
            with (out_dir / "curves" / f"{Path(path).stem}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch"] + keys)
                for e in sorted(epochs):
                    writer.writerow([e + 1] + [epochs[e].get(k, "") for k in keys])
        details.append("")
    doc += overview + [""] + details

    chains = sorted(glob.glob(os.path.join(args.log_dir, "qap-chain-*.log")),
                    key=lambda p: (int(JOB_ID.search(p).group(1)), p))
    if chains:
        doc += ["## 串行任务的进度", ""]
        for path in chains:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            doc += [f"### {Path(path).name}（{job_state(JOB_ID.search(path).group(1))}）", "", "```"]
            doc += [line for line in text.split("\n") if line.strip()][-20:] + ["```", ""]

    doc += ["## 评估任务的日志", ""]
    for path in sorted(glob.glob(os.path.join(args.log_dir, "eval-*.log")),
                       key=lambda p: (int(JOB_ID.search(p).group(1)), p))[-6:]:
        info = parse_log(path)
        text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        tail = [line for line in text.split("\n") if line.strip()][-25:]
        doc += [f"### {Path(path).name}（{job_state(info['job_id'])}）", "", "```"] + tail + ["```", ""]

    scans = sorted(glob.glob(os.path.join(args.log_dir, "act-scan-*.log")),
                   key=lambda p: (int(JOB_ID.search(p).group(1)), p))[-3:]
    if scans:
        doc += ["## 激活量化扫描的日志", ""]
        for path in scans:
            text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
            lines = [line for line in text.split("\n") if line.strip()]
            doc += [f"### {Path(path).name}（{job_state(JOB_ID.search(path).group(1))}）", "", "```"] + lines[-260:] + ["```", ""]

    (out_dir / "report.md").write_text("\n".join(doc) + "\n", encoding="utf-8")
    print(f"已写出 {out_dir / 'report.md'}（{len(logs)} 个训练任务）")


if __name__ == "__main__":
    main()
