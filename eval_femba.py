"""评估一个 FEMBA 分类 checkpoint，保存原始输出和全部指标。

和训练代码保持一致的部分：模型结构和配置从 checkpoint 里读取，输入归一化用任务里的同一个函数，
微调时不做掩码。指标全部在这里用标准定义重新计算，不依赖训练代码里的指标。

默认只在验证集上评估。测试集要显式写 --split test：
所有的比较和选择都在验证集上做，测试集只给已经确定的最终配置各用一次。

用法（在 BioFoundation 目录下、GPU 节点上运行）：
  python eval_femba.py --checkpoint /path/to/model.ckpt --name fp32 --out-dir /path/to/results
  python eval_femba.py --checkpoint /path/to/model.ckpt --name fp32 --out-dir /path/to/results --split test

权重量化（权重 8 bit 或 4 bit，激活浮点）有两种用法：
  PTQ：给浮点 checkpoint 加上 --ptq-bits 8 或 4，读取后直接把权重量化再评估，不训练。
       --ptq-init 选步长怎么定：max（每个通道的最大值，默认）或 mse（让量化误差最小的截断点）。
       --ptq-w8 是 --ptq-bits 8 的旧写法，仍然可用。
  QAT / QAP：直接给带量化训练得到的 checkpoint，脚本会自动识别位宽并按同样的位置重建量化器
激活量化（models/act_quant.py）同理：
  PTQ：加上 --ptq-act-bits 8 或 4（可以和 --ptq-bits 一起用），范围用训练集里的窗口校准，
       --ptq-act-percentile 选分位数。
  QAT / QAP：checkpoint 里带有激活量化器时自动识别并重建。
量化的位置是 28 处权重：patch 卷积、每个 Mamba 的 in_proj/conv1d/x_proj/dt_proj/out_proj、
分类头的两个线性层。偏置、位置编码、LayerNorm、A_log、D 和所有激活保持浮点。

输出（都在 --out-dir 下，文件名带 name 和 split）：
  <name>_<split>_metrics.json      全部指标
  <name>_<split>_predictions.npz   每个窗口的 logits、标签、受试者、来源，之后想加指标不用重跑模型
  <name>_<split>_per_subject.csv   每个受试者的窗口数、正样本比例、准确率

指标说明：
  auroc / aupr          用 logit[1] - logit[0] 排序（等价于用"有伪迹"的概率），标准算法
  auroc_author_style    只用 logit[1] 排序，作者训练代码里的算法，仅供对照
  accuracy              以概率 0.5 为阈值
  balanced_accuracy     两类召回率的平均
  f1                    以"有伪迹"为正类
  ece                   期望校准误差，15 个等宽置信度区间
  loss                  交叉熵
  auroc_ci95            按受试者重采样 1000 次得到的 95% 区间
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# 指标：只用 numpy，不依赖 torch 和 scikit-learn（定义与 scikit-learn 相同，已对照验证）
# ----------------------------------------------------------------------------
def softmax_p1(logits):
    """二分类 logits [N, 2] -> 正类（有伪迹）的概率 [N]。"""
    diff = logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64)
    return 1.0 / (1.0 + np.exp(-diff))


def roc_auc_score(labels, score):
    """AUROC = 随机取一个正样本和一个负样本，正样本得分更高的概率（得分相同算一半）。"""
    labels = np.asarray(labels).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC 需要同时有正样本和负样本。")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    # 得分相同的样本取平均名次
    starts = np.flatnonzero(np.r_[True, sorted_score[1:] != sorted_score[:-1]])
    ends = np.r_[starts[1:], len(score)]
    mean_rank = (starts + ends + 1) / 2.0
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.repeat(mean_rank, ends - starts)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_score(labels, score):
    """AUPR（平均精确率）：阈值从高到低移动，每次召回率的增量乘以当时的精确率，再求和。"""
    labels = np.asarray(labels).astype(np.float64)
    score = np.asarray(score, dtype=np.float64)
    n_pos = labels.sum()
    if n_pos == 0:
        raise ValueError("AUPR 需要至少一个正样本。")
    order = np.argsort(-score, kind="mergesort")
    sorted_score, sorted_labels = score[order], labels[order]
    last_of_tie = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(score) - 1]
    tp = np.cumsum(sorted_labels)[last_of_tie]
    precision = tp / (last_of_tie + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def expected_calibration_error(p1, labels, n_bins=15):
    """ECE：模型的置信度和实际正确率之间的平均差距，按每个区间的样本数加权。"""
    pred = (p1 >= 0.5).astype(np.int64)
    conf = np.where(pred == 1, p1, 1.0 - p1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def compute_metrics(logits, labels, subjects=None, n_boot=1000, seed=0):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    score = logits[:, 1] - logits[:, 0]
    p1 = softmax_p1(logits)
    pred = (p1 >= 0.5).astype(np.int64)

    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    recall_pos = tp / max(tp + fn, 1)
    recall_neg = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    eps = 1e-12
    p_true = np.where(labels == 1, p1, 1.0 - p1)

    out = {
        "n_windows": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, score)),
        "aupr": float(average_precision_score(labels, score)),
        "auroc_author_style": float(roc_auc_score(labels, logits[:, 1])),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall_pos + recall_neg) / 2,
        "f1": 2 * precision * recall_pos / max(precision + recall_pos, eps),
        "precision": precision,
        "recall": recall_pos,
        "specificity": recall_neg,
        "ece": expected_calibration_error(p1, labels),
        "loss": float(-np.log(np.clip(p_true, eps, None)).mean()),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }

    if subjects is not None:
        subjects = np.asarray(subjects)
        uniq = np.unique(subjects)
        index_of = {s: np.flatnonzero(subjects == s) for s in uniq}
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([index_of[s] for s in rng.choice(uniq, size=len(uniq))])
            if 0 < labels[idx].sum() < len(idx):
                boots.append(roc_auc_score(labels[idx], score[idx]))
        out["n_subjects"] = int(len(uniq))
        if boots:
            out["auroc_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def per_subject_rows(logits, labels, subjects):
    p1 = softmax_p1(np.asarray(logits, dtype=np.float64))
    pred = (p1 >= 0.5).astype(np.int64)
    rows = []
    for s in np.unique(subjects):
        m = subjects == s
        rows.append([s, int(m.sum()), round(float(labels[m].mean()), 4),
                     round(float((pred[m] == labels[m]).mean()), 4)])
    return rows


# ----------------------------------------------------------------------------
# 模型推理
# ----------------------------------------------------------------------------
def read_ids(h5_path):
    """按 HDF5Loader 相同的顺序读出每个窗口的受试者和来源；旧数据没有这两项时返回 None。"""
    import h5py
    subjects, sources = [], []
    with h5py.File(h5_path, "r") as f:
        for key in list(f.keys()):
            grp = f[key]
            if "subject" not in grp or "source" not in grp:
                return None, None
            subjects.append(np.array([s.decode() for s in grp["subject"][:]]))
            sources.append(np.array([s.decode() for s in grp["source"][:]]))
    return np.concatenate(subjects), np.concatenate(sources)


def run_inference(checkpoint_path, split, batch_size, data_file=None, ptq_bits=0, ptq_init="max",
                  ptq_act_bits=0, ptq_act_percentile=99.999, calib_windows=1024):
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset

    from datasets.hdf5_dataset import HDF5Loader
    from models.weight_quant import add_weight_quant, check_weight_quant
    from tasks.finetune_task import FinetuneTask

    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda key: os.getenv(key))
    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU：请通过 sbatch 在 GPU 节点上运行。")

    # 这是自己训练得到的 checkpoint，里面带有配置对象，需要用完整模式读取
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["hyper_parameters"])
    state = checkpoint["state_dict"]
    is_qat = any(".parametrizations.weight." in key for key in state)
    has_act = any(key.startswith("model.act_quant.") for key in state)
    trained_quant = is_qat or has_act
    if trained_quant and (ptq_bits or ptq_act_bits):
        raise ValueError("这个 checkpoint 已经是带量化训练得到的，不要再加 PTQ 选项。")
    task = FinetuneTask(cfg)
    n_quant, quant_error, bits, act_bits, n_act = 0, None, 0, 0, 0
    if is_qat:
        # 带量化训练的 checkpoint：先按同样的位置装上量化器，再读取权重和量化步长
        bits = int(cfg.get("qap_weight_bits", 8))  # 早先的 W8A32 checkpoint 没有记录位宽，是 8
        n_quant = len(add_weight_quant(task.model, bits))
    if has_act:
        # 带激活量化训练的 checkpoint：装上激活量化器，范围和开关从 checkpoint 里读
        from models.act_quant import install_act_quant
        act_bits = int(cfg.get("qap_act_bits", 0))
        if act_bits not in (4, 8):
            raise ValueError("checkpoint 里有激活量化器，但配置里没有记录激活位宽。")
        n_act = len(install_act_quant(task.model, act_bits).q)
    task.load_state_dict(state, strict=True)  # 有一处对不上就报错
    if has_act and task.model.act_quant.count_enabled() != n_act:
        raise RuntimeError("checkpoint 里有激活量化器没有打开。")
    if ptq_bits:
        # PTQ：读完浮点权重后再装量化器，步长直接由权重决定，不训练
        bits = int(ptq_bits)
        n_quant = len(add_weight_quant(task.model, bits, init=ptq_init))
    if n_quant:
        _, quant_error = check_weight_quant(task.model)  # 同时确认量化后的取值都在允许的等级上
    task.eval().cuda()

    def forward(X):
        X = X.cuda()
        if getattr(task, "normalize", False):
            X = task.normalize_fct(X)
        mask = torch.zeros(X.shape, dtype=torch.bool, device=X.device)  # 微调和评估时不遮挡
        return task.model(X, mask)[0]

    if ptq_act_bits:
        # 激活的 PTQ：用训练集里随机抽的窗口校准范围（固定种子），不训练
        from models.act_quant import install_act_quant
        act_bits = int(ptq_act_bits)
        aq = install_act_quant(task.model, act_bits)
        n_act = len(aq.q)
        train_set = HDF5Loader(str(cfg.data_module["train"].hdf5_file), use_cache=False)
        picked = sorted(np.random.default_rng(0).choice(
            len(train_set), size=min(calib_windows, len(train_set)), replace=False).tolist())
        calib = [X for X, _ in DataLoader(Subset(train_set, picked), batch_size=128, shuffle=False, num_workers=0)]

        def run():
            for X in calib:
                forward(X)

        from models.act_quant import calibrate_sequential
        with torch.no_grad():
            calibrate_sequential(aq, run, act_bits, ptq_act_percentile)  # 按前向顺序逐个位置校准

    w, a = (f"w{bits}" if bits else "w32"), (f"a{act_bits}" if act_bits else "a32")
    if trained_quant:
        quant = f"{w}{a}_qat"
    elif ptq_bits or ptq_act_bits:
        quant = f"{w}{a}_ptq" + (f"_{ptq_init}" if ptq_bits else "") + (f"_p{ptq_act_percentile:g}" if ptq_act_bits else "")
    else:
        quant = "fp32"
    print(f"量化方式：{quant}；被量化的权重张量 {n_quant} 个"
          + (f"；相对量化误差 {quant_error:.4f}" if n_quant else "")
          + (f"；激活量化器 {n_act} 个" if n_act else ""))

    h5_path = data_file or str(cfg.data_module[split].hdf5_file)
    dataset = HDF5Loader(h5_path, use_cache=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            all_logits.append(forward(X).float().cpu().numpy())
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels).astype(np.int64)

    subjects, sources = read_ids(h5_path)
    if subjects is not None and len(subjects) != len(labels):
        raise RuntimeError("受试者信息和样本数量对不上。")
    info = {"checkpoint": str(checkpoint_path), "data_file": h5_path,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "quant": quant, "weight_bits": bits, "n_quantized_weight_tensors": n_quant,
            "weight_quant_relative_error": quant_error, "act_bits": act_bits, "n_act_quantizers": n_act,
            "act_percentile": ptq_act_percentile if ptq_act_bits else None}
    return logits, labels, subjects, sources, info


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True, help="这个模型的名字，用在输出文件名里，如 fp32")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-file", default=None,
                        help="可选：直接指定 h5 文件；默认用 checkpoint 配置里对应 split 的路径")
    parser.add_argument("--ptq-bits", type=int, choices=[4, 8], default=0,
                        help="把浮点 checkpoint 的权重量化到这个位宽后评估（不训练）")
    parser.add_argument("--ptq-init", choices=["max", "mse"], default="max",
                        help="PTQ 的步长怎么定：max = 每个通道的最大值，mse = 量化误差最小的截断点")
    parser.add_argument("--ptq-w8", action="store_true", help="等同于 --ptq-bits 8（旧写法）")
    parser.add_argument("--ptq-act-bits", type=int, choices=[4, 8], default=0,
                        help="把激活也量化到这个位宽后评估（不训练，范围用训练集校准）")
    parser.add_argument("--ptq-act-percentile", type=float, default=99.999,
                        help="激活 PTQ 的范围取哪个分位数：99、99.9、99.99、99.999 或 100")
    parser.add_argument("--calib-windows", type=int, default=1024, help="激活 PTQ 校准用多少个训练窗口")
    args = parser.parse_args()
    if args.ptq_w8:
        if args.ptq_bits not in (0, 8):
            parser.error("--ptq-w8 和 --ptq-bits 不一致")
        args.ptq_bits = 8

    logits, labels, subjects, sources, info = run_inference(
        args.checkpoint, args.split, args.batch_size, args.data_file, args.ptq_bits, args.ptq_init,
        args.ptq_act_bits, args.ptq_act_percentile, args.calib_windows)
    metrics = compute_metrics(logits, labels, subjects)
    metrics.update(info)
    metrics.update({"name": args.name, "split": args.split})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_{args.split}"
    (args.out_dir / f"{stem}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = {"logits": logits.astype(np.float32), "labels": labels}
    if subjects is not None:
        saved.update({"subject": subjects, "source": sources})
        with (args.out_dir / f"{stem}_per_subject.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["subject_id", "n_windows", "positive_rate", "accuracy"])
            w.writerows(per_subject_rows(logits, labels, subjects))
    np.savez_compressed(args.out_dir / f"{stem}_predictions.npz", **saved)

    print(f"\n模型 {args.name}，{args.split} 集，{metrics['n_windows']} 个窗口"
          + (f"，{metrics['n_subjects']} 个受试者" if "n_subjects" in metrics else ""))
    print(f"checkpoint：第 {info['checkpoint_epoch'] + 1} 轮，第 {info['checkpoint_global_step']} 步；"
          f"量化方式 {info['quant']}")
    for key in ("auroc", "aupr", "accuracy", "balanced_accuracy", "f1", "ece", "loss",
                "auroc_author_style"):
        print(f"  {key:20s} {metrics[key]:.6f}")
    if "auroc_ci95" in metrics:
        print(f"  {'auroc_ci95':20s} [{metrics['auroc_ci95'][0]:.4f}, {metrics['auroc_ci95'][1]:.4f}]")
    print(f"结果保存在 {args.out_dir}/{stem}_*")


if __name__ == "__main__":
    main()
