"""在一个微调好的浮点模型上做激活量化的 PTQ 扫描（不训练，只用验证集）。

回答四个问题：
  1. 重写的前向对不对：量化全关时和原模型是否一致。
  2. 所有位置用同一个分位数时，取哪个最好；8 bit、4 bit 各掉多少。
  3. 输入信号要不要单独用一个分位数（输入 × 其余位置的两两组合）。
  4. 哪一类位置最敏感：每次只量化一类；以及每次只放过一类（其余全量化）。

做法：
  - 校准用训练集里随机抽的窗口（默认 1024 个），评估用整个验证集。测试集不参与。
  - 校准按前向顺序逐个位置进行：校准某个位置时，前面的量化器已经打开（见 models/act_quant.py）。
  - 默认权重保持浮点，看到的变化全部来自激活量化。--weight-bits 8 或 4 可以叠加权重量化（PTQ）。
  - 指标的算法和 eval_femba.py 相同。

输出（--out-dir 下）：act_scan_<名字>.csv，每种配置一行；日志里打印校准范围和汇总表。

用法（在 BioFoundation 目录下、GPU 节点上）：
  python scan_act_quant.py --checkpoint /path/to/fp.ckpt --name fp_ref --out-dir /path/to/results
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

from eval_femba import compute_metrics

KEYS = ("auroc", "aupr", "balanced_accuracy", "f1", "ece", "loss")
GROUP_LABEL = {
    "input": "输入信号", "embed": "embedding：卷积/位置编码/相加", "m_in_proj": "Mamba ① in_proj 输出",
    "m_conv": "Mamba ② 卷积输出", "m_silu": "Mamba ③ SiLU 输出（非对称）", "m_ssm": "Mamba ④ SSM 输出",
    "m_out": "Mamba ⑤ out_proj 输出", "merge": "正反向相加之后", "norm": "LayerNorm 之后",
    "head_fc1": "分类头 fc1 输出", "head_gelu": "分类头 GELU 输出（非对称）"}


def load_task(checkpoint_path):
    import torch
    from omegaconf import OmegaConf
    from tasks.finetune_task import FinetuneTask

    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver("env", lambda key: os.getenv(key))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    if any(".parametrizations.weight." in key for key in state):
        raise ValueError("这个脚本要用浮点模型的 checkpoint，这个 checkpoint 是带量化训练的。")
    cfg = OmegaConf.create(checkpoint["hyper_parameters"])
    task = FinetuneTask(cfg)
    task.load_state_dict(state, strict=True)
    return task.eval().cuda(), cfg


def load_windows(h5_path, count=None, seed=0):
    """读出窗口和标签。count 给定时随机抽这么多个（固定种子），否则读全部。"""
    import torch
    from torch.utils.data import DataLoader, Subset
    from datasets.hdf5_dataset import HDF5Loader

    dataset = HDF5Loader(h5_path, use_cache=False)
    if count is not None and count < len(dataset):
        rng = np.random.default_rng(seed)
        dataset = Subset(dataset, sorted(rng.choice(len(dataset), size=count, replace=False).tolist()))
    xs, ys = [], []
    for X, y in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0):
        xs.append(X)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys).numpy().astype(np.int64)


def predict(task, X, batch_size):
    import torch
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = X[start:start + batch_size].cuda()
            if getattr(task, "normalize", False):
                x = task.normalize_fct(x)
            mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
            outputs.append(task.model(x, mask)[0].float().cpu())
    return torch.cat(outputs).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calib-windows", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-bits", type=int, choices=[0, 4, 8], default=0,
                        help="同时把权重量化到这个位宽（PTQ）；0 = 权重保持浮点")
    parser.add_argument("--weight-init", choices=["max", "mse"], default="max")
    parser.add_argument("--quick", action="store_true", help="只做检查、统一分位数的扫描和每次只量化一类的敏感度")
    args = parser.parse_args()

    import torch
    from models.act_quant import GROUPS, PERCENTILES, calibrate_sequential, install_act_quant

    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU：请通过 sbatch 运行。")
    task, cfg = load_task(args.checkpoint)
    model = task.model
    if args.weight_bits:
        from models.weight_quant import add_weight_quant
        n = len(add_weight_quant(model, args.weight_bits, init=args.weight_init))
        print(f"权重：{args.weight_bits} bit（{n} 个张量，初始化 {args.weight_init}）")
    else:
        print("权重：浮点")

    X_val, y_val = load_windows(str(cfg.data_module["val"].hdf5_file))
    X_cal, _ = load_windows(str(cfg.data_module["train"].hdf5_file), args.calib_windows)
    print(f"验证集 {len(X_val)} 个窗口；校准用训练集里的 {len(X_cal)} 个窗口")

    # 1. 原模型
    logits_ref = predict(task, X_val, args.batch_size)
    base = compute_metrics(logits_ref, y_val)
    print(f"原模型：AUROC {base['auroc']:.4f}，平衡准确率 {base['balanced_accuracy']:.4f}，损失 {base['loss']:.4f}")

    # 2. 装上量化位置。全部关闭 -> 调用原来的模块，必须完全相同
    aq = install_act_quant(model, 8)
    print(f"激活量化器 {len(aq.q)} 个")
    diff_off = float(np.abs(predict(task, X_val, args.batch_size) - logits_ref).max())
    print(f"检查一：量化全部关闭，输出和原模型的最大差 = {diff_off:.3g}（应为 0）")
    if diff_off > 1e-6:
        raise RuntimeError("装上量化位置后，关闭状态下的输出变了，停止。")

    # 3. 全部关闭但强制走重写的 Mamba 路径：和原模型只应有浮点舍入量级的差别
    aq.force_custom = True
    start = time.time()
    logits_custom = predict(task, X_val, args.batch_size)
    seconds = time.time() - start
    aq.force_custom = False
    diff_custom = float(np.abs(logits_custom - logits_ref).max())
    custom = compute_metrics(logits_custom, y_val)
    print(f"检查二：量化全部关闭、走重写的前向，输出和原模型的最大差 = {diff_custom:.3g}；"
          f"AUROC {custom['auroc']:.6f}（原模型 {base['auroc']:.6f}）；一次验证集推理 {seconds:.1f} 秒")
    if diff_custom > 1e-2 or abs(custom["auroc"] - base["auroc"]) > 1e-4:
        raise RuntimeError("重写的前向和原模型不一致，后面的结果不可信，停止。")

    # 4. 先看浮点模型上各位置的数值范围（只用来了解分布；真正的校准在下面按前向顺序逐个进行）
    X_cal_batches = [X_cal[i:i + args.batch_size] for i in range(0, len(X_cal), args.batch_size)]

    def run_calib():
        for xb in X_cal_batches:
            x = xb.cuda()
            if getattr(task, "normalize", False):
                x = task.normalize_fct(x)
            model(x, torch.zeros(x.shape, dtype=torch.bool, device=x.device))

    for q in aq.q.values():
        q.calibrating = True
    with torch.no_grad():
        run_calib()
    print("\n浮点模型上各位置的分位数（对称的位置是绝对值，非对称的位置是数值本身；各批平均）：")
    print(f"{'量化器':<16}{'来源':<6}" + "".join(f"{p:>13g}%" for p in PERCENTILES))
    for name, q in aq.q.items():
        for key, (total, count) in q.stats.items():
            print(f"{name:<19}{key:<8}" + "".join(f"{float(v) / count:>14.4g}" for v in total))
        q.calibrating, q.stats = False, {}

    rows = [dict(bits=0, config="原模型（浮点）", pct_input="", pct_other="", group="",
                 **{k: base[k] for k in KEYS})]
    seq_ranges = {}

    def run(bits, config, pct, groups=None, pct_input=None, group=""):
        by_group = {"input": pct_input} if pct_input is not None else None
        with torch.no_grad():
            calibrate_sequential(aq, run_calib, bits, pct, groups=groups, percentile_by_group=by_group)
        if groups is None and pct_input is None:
            seq_ranges[bits, pct] = aq.ranges()
        m = compute_metrics(predict(task, X_val, args.batch_size), y_val)
        rows.append(dict(bits=bits, config=config, pct_input=pct if pct_input is None else pct_input,
                         pct_other=pct, group=group, **{k: m[k] for k in KEYS}))
        return m

    start = time.time()
    for bits in (8, 4):
        # 5. 所有位置同一个分位数
        uniform = {p: run(bits, "全部位置", p) for p in PERCENTILES}
        best = max(PERCENTILES, key=lambda p: uniform[p]["auroc"])
        for metric in ("auroc", "balanced_accuracy", "loss"):
            print(f"\n===== 激活 {bits} bit，全部位置量化，验证集 {metric}（原模型 {base[metric]:.4f}）=====")
            print("所有位置同一个分位数： " + "  ".join(f"{p:g}%: {uniform[p][metric]:.4f}" for p in PERCENTILES))
        print(f"统一分位数里 AUROC 最高的是 {best:g}%。这时各量化器的范围（按前向顺序校准得到）：")
        print("  " + "  ".join(f"{name}={value:.3g}" for name, value in seq_ranges[bits, best].items()))

        # 6. 逐类位置的敏感度：每次只量化一类
        alone = {g: run(bits, "只量化这一类", best, groups=[g], group=g) for g in GROUPS}
        print(f"\n===== 激活 {bits} bit，每次只量化一类位置（分位数 {best:g}%），验证集 =====")
        print(f"{'位置':<34}{'AUROC':>10}{'相对原模型':>12}{'平衡准确率':>12}")
        for g in GROUPS:
            print(f"{GROUP_LABEL[g]:<34}{alone[g]['auroc']:>12.4f}{alone[g]['auroc'] - base['auroc']:>+15.4f}"
                  f"{alone[g]['balanced_accuracy']:>15.4f}")
        if args.quick:
            continue

        # 7. 输入单独取分位数；每次只放过一类
        grid = {(pi, po): (uniform[po] if pi == po else run(bits, "全部位置", po, pct_input=pi))
                for pi in PERCENTILES for po in PERCENTILES}
        print(f"\n===== 激活 {bits} bit，输入单独取分位数，验证集 AUROC（行 = 输入的分位数，列 = 其余位置的分位数）=====")
        print(f"{'':>10}" + "".join(f"{p:>10g}%" for p in PERCENTILES))
        for pi in PERCENTILES:
            print(f"{pi:>9g}%" + "".join(f"{grid[pi, po]['auroc']:>11.4f}" for po in PERCENTILES))
        without = {g: run(bits, "只放过这一类", best, groups=[x for x in GROUPS if x != g], group=g) for g in GROUPS}
        print(f"\n===== 激活 {bits} bit，每次只放过一类位置（分位数 {best:g}%），验证集 AUROC =====")
        print(f"全部量化 {uniform[best]['auroc']:.4f}")
        for g in GROUPS:
            w = without[g]["auroc"]
            print(f"{GROUP_LABEL[g]:<34}{w:>12.4f}{w - uniform[best]['auroc']:>+15.4f}")
    print(f"\n扫描用时 {time.time() - start:.0f} 秒")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_w{args.weight_bits}{args.weight_init}" if args.weight_bits else ""
    path = args.out_dir / f"act_scan_{args.name}{suffix}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n共 {len(rows)} 种配置，结果保存在 {path}")


if __name__ == "__main__":
    main()
