"""FEMBA 的激活量化：量化器，以及整个模型上的 32 个量化位置。

量化器（ActQuantizer）
  - 整个张量一个步长。两种形式：
      对称    整数范围 [-Q, Q]，Q = 2^(位宽-1) - 1，零点为 0。
      非对称  整数范围 [0, 2^位宽 - 1]，下限固定（激活函数的最小值），上限可学习，
              零点 = round(-下限 / 步长)。用在 SiLU、GELU 的输出上。
  - 范围的初值靠校准：让一批训练数据流过模型，记录这个位置的分位数（对称记绝对值，
    非对称记数值本身的上侧），各批取平均。
  - 校准按前向的先后顺序逐个位置进行（calibrate_sequential）：校准某个位置时，它前面的量化器
    已经打开。这一点对这个模型是必须的：输入里的大幅值伪迹经过第一层 SSM 后会被放大到上百万
    （B、C 都随输入变化，输出大致按输入的三次方增长），在浮点模型上一次性校准所有位置，
    第一层的范围会被这些极端值撑开几个数量级。输入先被截断之后，后面的范围才是实际会遇到的。
  - 范围是可学习参数（存它的对数）。PTQ 不训练，停在初值；QAT、QAP 继续学习。
  - 梯度同权重量化器（白皮书式 37、38）：先截断再取整，STE 只作用于取整。

量化位置（详见项目文档 activation-quant-design）
  embedding 层   输入；卷积输出与位置编码（共用步长）；相加之后
  每个 Mamba     ① in_proj 输出  ② 卷积输出  ③ SiLU 输出（非对称）  ④ SSM 输出  ⑤ out_proj 输出
  编码器每层     正向、反向的 ⑤ 共用步长；相加之后（merge）；LayerNorm 之后（norm）
  分类头         fc1 输出；GELU 输出（非对称）；Mamba 的 ①-⑤；时间平均之后沿用 ⑤ 的步长
  保持浮点       x_proj 的输出、dt、B、C、递推的隐状态、z 路的 SiLU 和门控乘法；残差相加；
                 LayerNorm 的计算；最后的两个得分；翻转前后不加量化

怎么装到模型上
  mamba_ssm 的 Mamba 前向是封装好的，中间结果拿不到。这里按它的非融合路径（mamba_ssm 2.2.6
  的 mamba_simple.py）把这几步重写一遍并插入量化器，递推仍然调用它的 selective_scan_fn。
  作者的 FEMBA.forward、MambaWrapper.forward、MambaClassifier.forward 也各重写一遍，
  然后替换这个模型实例的 forward；作者的文件不改动。
  某一部分的量化器全部关闭时，这一部分直接调用原来的模块，所以结果和原模型逐位相同。
  重写的路径本身是否正确，由 scan_act_quant.py 检查（量化全关、强制走重写路径，和原模型比较）。

自检（只检查量化器本身，CPU 即可）：python -m models.act_quant
"""
import math
import types

import torch
import torch.nn.functional as F
from torch import nn

PERCENTILES = (95.0, 99.0, 99.9, 99.99, 99.999, 100.0)  # 100 = 最大值
SILU_MIN = -0.2784645427610738   # x * sigmoid(x) 的最小值，在 x ≈ -1.2785 处
GELU_MIN = -0.16997120747990106  # x * Phi(x) 的最小值，在 x ≈ -0.7518 处
GROUPS = ("input", "embed", "m_in_proj", "m_conv", "m_silu", "m_ssm", "m_out",
          "merge", "norm", "head_fc1", "head_gelu")
_MAX_SORT = 8_000_000  # 校准时一次最多排序这么多个数，更大的张量等间隔抽取


class ActQuantizer(nn.Module):
    def __init__(self, bits=8, lower=None):
        super().__init__()
        self.lower = None if lower is None else float(lower)  # None = 对称
        self.log_range = nn.Parameter(torch.zeros(()))  # 对称：最大绝对值；非对称：上限 - 下限
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("enabled", torch.tensor(False))
        self.set_bits(bits)
        self.calibrating = False
        self.stats = {}  # 来源名字 -> [各分位数的累加值, 批数]

    def set_bits(self, bits):
        if bits not in (8, 4):
            raise ValueError(f"位宽只支持 8 或 4，收到 {bits}。")
        self.bits = int(bits)
        self.levels = (2 ** (self.bits - 1) - 1) if self.lower is None else (2 ** self.bits - 1)

    # ---------------- 校准 ----------------
    @torch.no_grad()
    def observe(self, x, key):
        values = x.detach().flatten().float()
        if self.lower is None:
            values = values.abs()
        if values.numel() > _MAX_SORT:
            values = values[::math.ceil(values.numel() / _MAX_SORT)]
        values = values.sort().values
        n = values.numel()
        picked = torch.stack([values[min(n - 1, max(0, math.ceil(n * p / 100.0) - 1))]
                              for p in PERCENTILES]).double().cpu()
        total, count = self.stats.get(key, (torch.zeros(len(PERCENTILES), dtype=torch.float64), 0))
        self.stats[key] = (total + picked, count + 1)

    def calibrated_range(self, percentile):
        """校准得到的范围：每个来源取各批的平均，多个来源（共用步长时）取最大的。"""
        if not self.stats:
            raise RuntimeError("这个量化器还没有校准数据。")
        index = PERCENTILES.index(float(percentile))
        top = max(float(total[index] / count) for total, count in self.stats.values())
        return max(top if self.lower is None else top - self.lower, 1e-6)

    @torch.no_grad()
    def set_range(self, value):
        self.log_range.fill_(math.log(max(float(value), 1e-8)))
        self.initialized.fill_(True)

    # ---------------- 量化 ----------------
    def forward(self, x, key="x"):
        if self.calibrating:
            self.observe(x, key)
            return x
        if not bool(self.enabled):
            return x
        if not bool(self.initialized):
            raise RuntimeError("激活量化器还没有设定范围：先校准（calibrate_sequential）。")
        scale = self.log_range.exp() / self.levels
        factor = 1.0 / math.sqrt(x[0].numel() * self.levels)
        scale = (scale - scale * factor).detach() + scale * factor  # 步长的梯度按 LSQ 缩小
        if self.lower is None:
            clipped = (x / scale).clamp(-self.levels, self.levels)
            integer = clipped + (clipped.round() - clipped).detach()  # 前向取整，反向直通
            return integer * scale
        zero_point = (-self.lower / scale).round().detach()
        clipped = (x / scale + zero_point).clamp(0, self.levels)
        integer = clipped + (clipped.round() - clipped).detach()
        return (integer - zero_point) * scale


class FembaActQuant(nn.Module):
    """整个模型的激活量化器。self.q 按名字存放，self.group 给出每个量化器属于哪一类位置。"""

    def __init__(self, model, bits=8):
        super().__init__()
        self.q = nn.ModuleDict()
        self.group = {}
        self.n_layers = len(model.mamba_blocks)
        self.force_custom = False  # True：量化器全关时也走重写的路径（只用于检查）

        def add(name, group, lower=None):
            self.q[name] = ActQuantizer(bits, lower)
            self.group[name] = group

        def add_mamba(prefix):
            add(f"{prefix}_q1", "m_in_proj")
            add(f"{prefix}_q2", "m_conv")
            add(f"{prefix}_q3", "m_silu", SILU_MIN)
            add(f"{prefix}_q4", "m_ssm")

        add("input", "input")
        add("conv_pos", "embed")
        add("sum", "embed")
        for i in range(len(model.mamba_blocks)):
            add_mamba(f"L{i}_fwd")
            add_mamba(f"L{i}_rev")
            add(f"L{i}_out", "m_out")      # 正向、反向的 ⑤ 共用
            add(f"L{i}_merge", "merge")
            add(f"L{i}_norm", "norm")
        self.has_head = model.classifier is not None
        if self.has_head:
            add("head_fc1", "head_fc1")
            add("head_gelu", "head_gelu", GELU_MIN)
            add_mamba("head")
            add("head_out", "m_out")       # 分类头 Mamba 的 ⑤，时间平均之后也用它

    def four(self, prefix):
        return [self.q[f"{prefix}_q{k}"] for k in (1, 2, 3, 4)]

    def active(self, prefix):
        """名字以 prefix 开头的量化器里，有没有在工作的（校准中、已打开，或强制走重写路径）。"""
        if self.force_custom:
            return True
        return any(q.calibrating or bool(q.enabled) for name, q in self.q.items() if name.startswith(prefix))

    def stages(self):
        """按前向的先后顺序给出量化器的名字，每一组是可以同时校准的。"""
        out = [["input"], ["conv_pos"], ["sum"]]
        for i in range(self.n_layers):
            out += [[f"L{i}_fwd_q{k}", f"L{i}_rev_q{k}"] for k in (1, 2, 3, 4)]
            out += [[f"L{i}_out"], [f"L{i}_merge"], [f"L{i}_norm"]]
        if self.has_head:
            out += [["head_fc1"], ["head_gelu"], ["head_q1"], ["head_q2"], ["head_q3"], ["head_q4"], ["head_out"]]
        if sorted(n for stage in out for n in stage) != sorted(self.q):
            raise RuntimeError("校准顺序和量化器的名字对不上。")
        return out

    def disable(self):
        for q in self.q.values():
            q.enabled.fill_(False)

    def count_enabled(self):
        return sum(bool(q.enabled) for q in self.q.values())

    def ranges(self):
        return {name: float(q.log_range.exp()) for name, q in self.q.items() if bool(q.enabled)}


def calibrate_sequential(aq, run, bits, percentile, groups=None, percentile_by_group=None, prefix=""):
    """按前向顺序逐个位置校准并打开量化器。run() 让校准数据流过模型一遍（调用方负责 no_grad）。

    只处理名字以 prefix 开头的量化器：其中属于 groups（None = 全部类别）的被校准并打开，其余的关闭。
    名字不以 prefix 开头的量化器保持原样（QAP 微调时编码器的量化器已经从预训练里读入并打开）。
    percentile_by_group 给个别类别单独指定分位数，例如 {"input": 99.0}。返回打开了几个。
    """
    chosen = set(GROUPS if groups is None else groups)
    unknown = chosen - set(GROUPS)
    if unknown:
        raise ValueError(f"不认识的位置类别：{sorted(unknown)}")
    for name, q in aq.q.items():
        if name.startswith(prefix):
            q.stats, q.calibrating = {}, False
            q.enabled.fill_(False)
    count = 0
    for stage in aq.stages():
        todo = [name for name in stage if name.startswith(prefix) and aq.group[name] in chosen]
        if not todo:
            continue
        for name in todo:
            aq.q[name].calibrating = True
        run()
        for name in todo:
            q = aq.q[name]
            q.calibrating = False
            q.set_bits(bits)
            q.set_range(q.calibrated_range((percentile_by_group or {}).get(aq.group[name], percentile)))
            q.enabled.fill_(True)
            count += 1
    return count


def mamba_forward(m, hidden, q1, q2, q3, q4, q_out, key):
    """一个 Mamba 的前向，和 mamba_ssm 的非融合路径逐步对应，插入 ①-⑤。hidden: (B, L, D)。"""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    batch, seqlen, _ = hidden.shape
    xz = q1(m.in_proj(hidden))                               # ① (B, L, 2*d_inner)
    x, z = xz.transpose(1, 2).chunk(2, dim=1)                # 各 (B, d_inner, L)
    x = q2(m.conv1d(x)[..., :seqlen])                        # ② 卷积输出
    x = q3(F.silu(x))                                        # ③ SiLU 输出，也是 SSM 的输入
    x_dbl = m.x_proj(x.transpose(1, 2).reshape(batch * seqlen, -1))
    dt, B, C = torch.split(x_dbl, [m.dt_rank, m.d_state, m.d_state], dim=-1)
    dt = (m.dt_proj.weight @ dt.t()).view(-1, batch, seqlen).permute(1, 0, 2).contiguous()  # 偏置在递推里加
    B = B.view(batch, seqlen, m.d_state).transpose(1, 2).contiguous()
    C = C.view(batch, seqlen, m.d_state).transpose(1, 2).contiguous()
    A = -torch.exp(m.A_log.float())
    y = selective_scan_fn(x.contiguous(), dt, A, B, C, m.D.float(), z=z.contiguous(),
                          delta_bias=m.dt_proj.bias.float(), delta_softplus=True,
                          return_last_state=False)           # 递推和 z 路门控，全部浮点
    y = q4(y.transpose(1, 2))                                # ④ SSM 输出 (B, L, d_inner)
    return q_out(m.out_proj(y), key)                         # ⑤


def _encoder_layer(model, aq, i, x):
    block = model.mamba_blocks[i]
    if not (aq.active(f"L{i}_fwd") or aq.active(f"L{i}_rev") or aq.active(f"L{i}_out") or aq.active(f"L{i}_merge")):
        return block(x)
    shared = aq.q[f"L{i}_out"]
    out = mamba_forward(block.mamba_fwd, x, *aq.four(f"L{i}_fwd"), shared, "fwd")
    rev = mamba_forward(block.mamba_rev, x.flip(dims=(1,)), *aq.four(f"L{i}_rev"), shared, "rev").flip(dims=(1,))
    return aq.q[f"L{i}_merge"](out + rev)


def _classifier(model, aq, x):
    head = model.classifier
    if not aq.active("head"):
        return head(x)
    x = aq.q["head_fc1"](head.fc1(x))
    x = aq.q["head_gelu"](head.activation1(x))
    x = mamba_forward(head.mamba_1, x, *aq.four("head"), aq.q["head_out"], "out")
    x = x.permute(0, 2, 1).contiguous().mean(dim=-1)         # 沿时间取平均
    x = aq.q["head_out"](x, "pool")                          # 平均之后沿用 ⑤ 的步长再量化
    x = head.fc3(x)                                          # 最后的得分不量化
    if head.classification_type == "mmc":
        x = x.view(-1, head.num_channels, head.num_classes)
    return x


def _forward_with_act_quant(self, x, mask):
    """和作者的 FEMBA.forward 逐行对应。"""
    aq = self.act_quant
    x_original = x
    x_masked = x.clone()
    x_masked[mask] = 0
    x_masked = aq.q["input"](x_masked)                       # 输入信号
    x = self.patch_embed(x_masked)
    x = aq.q["conv_pos"](x, "conv")                          # 卷积输出
    x = x + aq.q["conv_pos"](self.pos_embed, "pos")          # 位置编码，共用步长
    x = aq.q["sum"](x)                                       # 相加之后

    for i, norm_layer in enumerate(self.norm_layers):
        res = x
        x = _encoder_layer(self, aq, i, x)
        x = res + x                                          # 残差相加，浮点
        x = norm_layer(x)
        x = aq.q[f"L{i}_norm"](x)                            # LayerNorm 之后

    if self.classifier is not None:
        return _classifier(self, aq, x), x_original
    return self.decoder(x), x_original


def install_act_quant(model, bits=8):
    """给一个 FEMBA 模型实例装上全部激活量化位置，返回 FembaActQuant。

    装上后所有量化器都是关闭的，模型的输出和原来完全相同。
    之后用 calibrate_sequential(...) 校准并打开。
    """
    if hasattr(model, "act_quant"):
        raise RuntimeError("这个模型已经装过激活量化了。")
    for name in ("patch_embed", "pos_embed", "mamba_blocks", "norm_layers", "classifier"):
        if not hasattr(model, name):
            raise RuntimeError(f"模型里没有 {name}，不是预期的 FEMBA 结构。")
    for block in model.mamba_blocks:
        if not getattr(block, "bidirectional", False) or block.bidirectional_strategy != "add":
            raise RuntimeError("只支持双向、相加合并的 Mamba 块。")
    aq = FembaActQuant(model, bits).to(model.pos_embed.device)
    expected = 3 + 11 * len(model.mamba_blocks) + (7 if aq.has_head else 0)
    if len(aq.q) != expected:
        raise RuntimeError(f"应该有 {expected} 个激活量化器，实际 {len(aq.q)} 个。")
    model.act_quant = aq
    model.forward = types.MethodType(_forward_with_act_quant, model)
    return aq


def calibrate_in_training(module, aq, bits, percentile, run_batch, loader, n_batches, prefix=""):
    """训练开始前的校准：取训练集的前 n_batches 批，按前向顺序给（名字以 prefix 开头的）量化器定范围并打开。

    run_batch(batch) 负责把一批数据送进模型。校准不更新任何权重。
    校准前后保存并恢复随机数状态，这样带激活量化的训练和不带的训练，之后的数据顺序、遮挡位置仍然相同。
    返回 (用了几批, 打开了几个量化器)。
    """
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = module.training
    module.eval()
    batches = []
    for batch in loader:
        if len(batches) >= n_batches:
            break
        batches.append(batch)

    def run():
        for batch in batches:
            run_batch(batch)

    with torch.no_grad():
        count = calibrate_sequential(aq, run, bits, percentile, prefix=prefix)
    module.train(was_training)
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return len(batches), count


def _self_test():
    torch.manual_seed(0)

    # 1. 对称：4 bit，范围 0.7 -> 步长 0.1
    q = ActQuantizer(4)
    q.set_range(0.7)
    q.enabled.fill_(True)
    x = torch.tensor([0.34, -0.12, 5.0, -3.0], requires_grad=True)
    y = q(x)
    assert torch.allclose(y, torch.tensor([0.3, -0.1, 0.7, -0.7]), atol=1e-6), y
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([1.0, 1.0, 0.0, 0.0])), x.grad
    assert torch.isfinite(q.log_range.grad)
    q.set_bits(8)                                   # 换位宽后范围不变，格子变细
    assert abs(q(torch.tensor([0.7])).item() - 0.7) < 1e-6
    q.enabled.fill_(False)
    assert torch.equal(q(x), x)

    # 2. 非对称：4 bit，下限 -0.25，上限 3.5 -> 步长 0.25，零点 1，能表示 -0.25 到 3.5
    q = ActQuantizer(4, lower=-0.25)
    q.set_range(3.75)
    q.enabled.fill_(True)
    x = torch.tensor([0.0, -0.25, 0.30, 3.40, 9.0, -2.0], requires_grad=True)
    y = q(x)
    assert torch.allclose(y, torch.tensor([0.0, -0.25, 0.25, 3.5, 3.5, -0.25]), atol=1e-6), y
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])), x.grad
    levels = torch.unique(q(torch.linspace(-1, 5, 20001)))
    assert levels.numel() == 16 and abs(levels.min().item() + 0.25) < 1e-6 and abs(levels.max().item() - 3.5) < 1e-6
    silu = F.silu(torch.linspace(-20, 20, 400001, dtype=torch.float64))
    gelu = F.gelu(torch.linspace(-20, 20, 400001, dtype=torch.float64))
    assert abs(silu.min().item() - SILU_MIN) < 1e-6 and abs(gelu.min().item() - GELU_MIN) < 1e-6

    # 3. 校准：分位数不减，100 等于最大值；共用时取较大的来源；非对称记的是数值本身
    q = ActQuantizer(8)
    q.calibrating = True
    a = torch.randn(4, 1000)
    a[0, 0] = -50.0
    assert q(a, "fwd") is a
    q(torch.full((10,), 0.01), "rev")
    ranges = [q.calibrated_range(p) for p in PERCENTILES]
    assert all(r1 <= r2 + 1e-12 for r1, r2 in zip(ranges, ranges[1:])), ranges
    assert abs(ranges[-1] - 50.0) < 1e-5 and ranges[0] < 5.0, ranges
    q = ActQuantizer(8, lower=SILU_MIN)
    q.calibrating = True
    q(F.silu(a))
    assert abs(q.calibrated_range(100.0) - (F.silu(a).max().item() - SILU_MIN)) < 1e-5
    print("自检通过（量化器）。整个模型上的检查在 scan_act_quant.py 里，需要 GPU。")


if __name__ == "__main__":
    _self_test()
