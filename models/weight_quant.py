"""权重量化器：位宽可选（8 bit 或 4 bit），PTQ、QAT、QAP 共用这一个。

只量化权重，激活保持浮点。做法：
  - 每个输出通道一个量化步长（per-channel），对称，零点为 0。
  - 整数范围是 [-qmax, qmax]，qmax = 2^(位宽-1) - 1：8 bit 是 ±127，4 bit 是 ±7。
  - 量化后的权重 = round(权重 / 步长) 截断到整数范围，再乘回步长。
  - 步长可以学习（LSQ 的做法）。训练时浮点权重照常更新，前向用的是量化后的值。
    落在范围内的权重，梯度直接传回去（直通估计）；被截断的权重不再得到梯度，
    它们的梯度改为作用在步长上。

步长的初始值有两种：
  max  每个通道权重绝对值的最大值 / qmax。不截断任何权重。作者（Brevitas 默认）用的是这种。
  mse  在"最大值的 30% 到 100%"之间找让量化误差最小的截断点。低位宽时，为了个别大权重
       而把步长设得很大会浪费等级，这种初始化更合理。

量化器用 torch 的 parametrization 装在 module.weight 上，所以模型里任何读取
module.weight 的地方（包括 Mamba 内部）拿到的都是量化后的权重。

量化哪些权重由 add_weight_quant 的 include / exclude 决定，默认是 FEMBA 的
patch 卷积、Mamba 块、分类头里所有的 Linear / Conv 权重；解码器和 LayerNorm 不量化。
偏置、位置编码、A_log、D 不是 Linear / Conv 的 weight，自然不会被量化。

自检（在 BioFoundation 目录下，CPU 即可）：python -m models.weight_quant
"""
import torch
from torch import nn
from torch.nn.utils import parametrize

SUPPORTED_BITS = (8, 4)


def _scale_gradient(x, factor):
    """前向不变，反向时把梯度乘以 factor。"""
    return (x - x * factor).detach() + x * factor


class WeightQuantizer(nn.Module):
    def __init__(self, out_channels, bits):
        super().__init__()
        if bits not in SUPPORTED_BITS:
            raise ValueError(f"位宽只支持 {SUPPORTED_BITS}，收到 {bits}。")
        self.bits = int(bits)
        self.qmax = 2 ** (self.bits - 1) - 1
        self.log_scale = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("enabled", torch.tensor(True))

    @torch.no_grad()
    def initialize(self, weight, init="max"):
        flat = weight.detach().reshape(weight.shape[0], -1)
        max_abs = flat.abs().amax(dim=1).clamp_min(1e-8)
        if init == "max":
            scale = max_abs / self.qmax
        elif init == "mse":
            best_error = torch.full_like(max_abs, float("inf"))
            scale = max_abs / self.qmax
            for ratio in torch.linspace(0.3, 1.0, 36).tolist():
                candidate = (max_abs * ratio / self.qmax).unsqueeze(1)
                quantized = (flat / candidate).round().clamp(-self.qmax, self.qmax) * candidate
                error = (quantized - flat).pow(2).sum(dim=1)
                better = error < best_error
                best_error = torch.where(better, error, best_error)
                scale = torch.where(better, candidate.squeeze(1), scale)
        else:
            raise ValueError(f"init 只支持 max 或 mse，收到 {init}。")
        self.log_scale.copy_(scale.log())
        self.initialized.fill_(True)

    def scale(self, weight):
        """步长，形状能和 weight 逐通道相乘。"""
        return self.log_scale.exp().view(weight.shape[0], *([1] * (weight.ndim - 1)))

    def forward(self, weight):
        if not bool(self.initialized):
            raise RuntimeError("量化器还没有初始化步长。")
        if not bool(self.enabled):
            return weight
        per_channel = weight.numel() // weight.shape[0]
        scale = _scale_gradient(self.scale(weight), 1.0 / (per_channel * self.qmax) ** 0.5)
        clipped = (weight / scale).clamp(-self.qmax, self.qmax)
        integer = clipped + (clipped.round() - clipped).detach()  # 前向取整，反向直通
        return integer * scale

    @torch.no_grad()
    def integers(self, weight):
        return (weight / self.scale(weight)).round().clamp(-self.qmax, self.qmax)


def add_weight_quant(model, bits, include=("patch_embed", "mamba_blocks", "classifier"),
                     exclude=("decoder", "norm"), init="max"):
    """给选中的 Linear / Conv 权重装上量化器，返回被量化的模块名字列表。

    步长按当前的权重初始化，所以要在权重准备好之后调用（随机初始化之后，或读入权重之后），
    并且要在建优化器之前调用，这样步长才会进入优化器。
    """
    selected = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            continue
        if not any(key in name for key in include) or any(key in name for key in exclude):
            continue
        if parametrize.is_parametrized(module, "weight"):
            raise RuntimeError(f"{name} 的权重已经装过量化器了。")
        selected.append((name, module))
    if not selected:
        raise RuntimeError("没有选中任何权重。")
    for name, module in selected:
        quantizer = WeightQuantizer(module.weight.shape[0], bits).to(module.weight.device)
        quantizer.initialize(module.weight, init)
        parametrize.register_parametrization(module, "weight", quantizer)
    return [name for name, _ in selected]


def iter_weight_quantizers(model):
    """逐个给出 (模块名, 模块, 量化器)。"""
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module, "weight"):
            for p in module.parametrizations.weight:
                if isinstance(p, WeightQuantizer):
                    yield name, module, p


@torch.no_grad()
def check_weight_quant(model):
    """确认每个被量化的权重确实只取允许的等级。返回 (张量个数, 相对量化误差)。

    相对量化误差 = 所有被量化权重的 ||量化后 - 原始|| / ||原始||。
    """
    count, error_sq, norm_sq = 0, 0.0, 0.0
    for name, module, q in iter_weight_quantizers(model):
        original = module.parametrizations.weight.original
        quantized = module.weight
        scale = q.scale(original)
        integer = quantized / scale
        if (integer - integer.round()).abs().max() > 1e-3:
            raise AssertionError(f"{name}：量化后的权重不是步长的整数倍。")
        if integer.abs().max() > q.qmax + 1e-3:
            raise AssertionError(f"{name}：量化后的权重超出了 ±{q.qmax}。")
        count += 1
        error_sq += (quantized - original).pow(2).sum().item()
        norm_sq += original.pow(2).sum().item()
    if count == 0:
        raise RuntimeError("模型里没有找到量化器。")
    return count, (error_sq / max(norm_sq, 1e-30)) ** 0.5


def _self_test():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(1, 4, (2, 4), stride=(2, 4))
            self.mamba_blocks = nn.ModuleList([nn.Linear(16, 16, bias=False)])
            self.classifier = nn.Linear(16, 2)
            self.decoder = nn.Linear(16, 16)

        def forward(self, x):
            return self.classifier(self.mamba_blocks[0](self.patch_embed(x).flatten(1)))

    errors = {}
    for bits in SUPPORTED_BITS:
        for init in ("max", "mse"):
            torch.manual_seed(0)
            model = Tiny()
            names = add_weight_quant(model, bits, init=init)
            assert names == ["patch_embed", "mamba_blocks.0", "classifier"], names
            assert not parametrize.is_parametrized(model.decoder, "weight")
            count, errors[bits, init] = check_weight_quant(model)
            assert count == 3
            qmax = 2 ** (bits - 1) - 1
            for _, module, q in iter_weight_quantizers(model):
                levels = max(torch.unique(row).numel() for row in q.integers(
                    module.parametrizations.weight.original).flatten(1))
                assert levels <= 2 * qmax + 1, levels
            model(torch.randn(3, 1, 4, 8)).square().mean().backward()
            for _, module, q in iter_weight_quantizers(model):
                assert module.parametrizations.weight.original.grad is not None
                assert q.log_scale.grad is not None and torch.isfinite(q.log_scale.grad).all()
            # 存下来再读回去，量化后的权重必须完全一样
            other = Tiny()
            add_weight_quant(other, bits)
            other.load_state_dict(model.state_dict(), strict=True)
            assert torch.equal(other.classifier.weight, model.classifier.weight)
            print(f"{bits} bit, init={init}: 相对量化误差 {errors[bits, init]:.4f}")
    assert errors[4, "max"] > errors[8, "max"] * 5, "4 bit 的误差应该明显大于 8 bit"
    assert errors[4, "mse"] <= errors[4, "max"] + 1e-9, "mse 初始化的误差不应该比 max 大"

    # 梯度：范围内的权重梯度为 1，被截断的权重梯度为 0，被截断处步长的梯度方向为 qmax
    q = WeightQuantizer(1, 4)
    w = torch.tensor([[0.30, -0.10, 5.0]], requires_grad=True)
    with torch.no_grad():
        q.log_scale.fill_(torch.tensor(0.1).log())
        q.initialized.fill_(True)
    out = q(w)
    assert torch.allclose(out, torch.tensor([[0.3, -0.1, 0.7]]), atol=1e-6), out
    out.sum().backward()
    assert torch.allclose(w.grad, torch.tensor([[1.0, 1.0, 0.0]])), w.grad
    assert q.log_scale.grad.item() > 0

    try:  # 8 bit 时前向结果应该和之前用的 qat_w8a32 完全一样
        from models.qat_w8a32 import apply_w8a32_fake_quant
        torch.manual_seed(0)
        a, b = Tiny(), Tiny()
        b.load_state_dict(a.state_dict())
        add_weight_quant(a, 8)
        apply_w8a32_fake_quant(b)
        for name in ("patch_embed", "classifier"):
            assert torch.allclose(getattr(a, name).weight, getattr(b, name).weight, atol=1e-7, rtol=0)
        assert set(a.state_dict()) == set(b.state_dict())
        print("8 bit 的前向结果和 checkpoint 格式与 qat_w8a32 一致")
    except ImportError:
        print("（没有找到 models.qat_w8a32，跳过和它的对比）")
    print("自检通过")


if __name__ == "__main__":
    _self_test()
