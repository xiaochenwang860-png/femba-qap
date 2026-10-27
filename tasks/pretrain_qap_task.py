"""带权重量化的掩码重建预训练（QAP），以及与它配对的浮点预训练。

预训练的做法和作者的 tasks/pretrain_task.py（MaskTask）相同：
  随机遮掉 60% 的小块（2 通道 x 16 点）-> 编码器 + 重建解码器 -> 只在被遮住的位置上算重建损失。

和作者任务的区别：
  1. 多一个开关 qap_weight_bits（在命令行用 +qap_weight_bits=8 给出）：
       0      浮点预训练（对照组）
       8 或 4  编码器的权重带 8 bit / 4 bit 量化预训练（QAP 组）
     另有 qap_act_bits（0、8 或 4）：编码器的激活也带量化（models/act_quant.py 的 25 个位置），
     范围在训练开始时用训练集的前几批、按 qap_act_percentile 分位数校准，之后可学习。
     量化的位置：patch 卷积 + 每个 Mamba 的 in_proj / conv1d / x_proj / dt_proj / out_proj。
     重建解码器不量化（它只在预训练时存在，不属于最终模型）。偏置、位置编码、LayerNorm、
     A_log、D 和所有激活保持浮点。量化器用 models/weight_quant.py，和微调阶段是同一个。
  2. 学习率调度由开关 qap_lr_schedule 决定（作者的 multi_step_lr 第一次衰减在 20 万步，
     我们的训练远到不了那里，照搬等同于全程固定，所以不用它）：
       cosine  先预热再余弦衰减（正式实验用）：前 qap_warmup_epochs 轮从接近 0 线性升到
               optimizer.lr，之后按余弦曲线降到接近 0。按步更新。
       const   全程固定为 optimizer.lr（只用于试跑，看损失多少轮走平）。
  3. 不画重建信号的对比图（那部分依赖 matplotlib）。
  4. 训练结束时把最后一轮的权重另存为 <输出目录>/final.ckpt。
     训练程序自带的 last.ckpt 并不是最后一轮：它只在验证损失创新低时才更新，内容等于
     验证损失最低的那一轮。各组要用相同的训练量比较，所以微调一律从 final.ckpt 出发。

两组实验除了 qap_weight_bits 之外的设置必须完全相同，包括随机种子，这样两次的初始权重也相同。
"""
import math
import os

import hydra
import pytorch_lightning as pl
import torch
import torch_optimizer as torch_optim

from biofoundation.core.batch import as_signal_batch
from util.train_utils import RobustQuartileNormalize


class QAPPretrainTask(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(self.hparams.model)
        if getattr(self.model, "classifier", None) is not None:
            raise ValueError("预训练需要 model.num_classes=0（带重建解码器，不带分类头）。")
        self.criterion = hydra.utils.instantiate(self.hparams.criterion)
        self.patch_size = self.hparams.masking.patch_size
        self.masking_ratio = self.hparams.masking.masking_ratio

        self.normalize = False
        norm = self.hparams.input_normalization
        if norm is not None and norm.normalize:
            self.normalize = True
            self.normalize_fct = RobustQuartileNormalize(
                norm.quartile_normalization_lower_val, norm.quartile_normalization_upper_val)

        self.weight_bits = int(self.hparams.get("qap_weight_bits", 0))
        if self.weight_bits not in (0, 8, 4):
            raise ValueError(f"qap_weight_bits 只支持 0（浮点）、8 或 4，收到 {self.weight_bits}。")
        self.act_bits = int(self.hparams.get("qap_act_bits", 0))
        if self.act_bits not in (0, 8, 4):
            raise ValueError(f"qap_act_bits 只支持 0（浮点）、8 或 4，收到 {self.act_bits}。")
        self.act_percentile = float(self.hparams.get("qap_act_percentile", 99.999))
        self._quant_ready = False

        self.lr_schedule = str(self.hparams.get("qap_lr_schedule", "cosine"))
        if self.lr_schedule not in ("cosine", "const"):
            raise ValueError(f"qap_lr_schedule 只支持 cosine 或 const，收到 {self.lr_schedule}。")
        self.warmup_epochs = float(self.hparams.get("qap_warmup_epochs", 2))

    # ------------------------------------------------------------------
    # 量化器的安装
    # ------------------------------------------------------------------
    def _install_quantizers(self):
        """在建优化器之前调用一次。浮点对照组什么都不装。"""
        if self._quant_ready:
            return
        if self.weight_bits > 0:
            from models.weight_quant import add_weight_quant, check_weight_quant
            names = add_weight_quant(self.model, self.weight_bits, include=("patch_embed", "mamba_blocks"))
            expected = 1 + 10 * int(self.hparams.model.num_blocks)
            if len(names) != expected:
                raise RuntimeError(f"应该量化 {expected} 个编码器权重，实际 {len(names)} 个。")
            if any("decoder" in name for name in names):
                raise RuntimeError("解码器不应该被量化。")
            _, error = check_weight_quant(self.model)
            print(f"[QAP] 编码器权重带 {self.weight_bits} bit 量化预训练：{len(names)} 个权重张量；"
                  f"解码器保持浮点；初始相对量化误差 {error:.4f}")
        if self.act_bits > 0:
            # 激活量化：编码器的 25 个位置。装上时是关闭的，范围在训练开始时校准（见 on_train_start）
            from models.act_quant import install_act_quant
            aq = install_act_quant(self.model, self.act_bits)
            print(f"[QAP] 激活带 {self.act_bits} bit 量化预训练：{len(aq.q)} 个量化器；"
                  f"范围在训练开始时用 {self.act_percentile:g}% 分位数校准")
        if self.weight_bits == 0 and self.act_bits == 0:
            print("[QAP] 浮点预训练（对照组）：不量化")
        self._quant_ready = True

    def on_train_start(self):
        if self.act_bits == 0:
            return
        from models.act_quant import calibrate_in_training

        def run_batch(batch):
            X = as_signal_batch(batch)["input"].to(self.device)
            mask = self.generate_mask(X.shape[0], X.shape[1], X.shape[2])
            if self.normalize:
                X = self.normalize_fct(X)
            self.model(X, mask)

        seen, count = calibrate_in_training(
            self, self.model.act_quant, self.act_bits, self.act_percentile, run_batch,
            self.trainer.datamodule.train_dataloader(), n_batches=4)
        print(f"[QAP] 激活量化已校准并打开：{count} 个量化器，用了训练集的 {seen} 批"
              f"（此时模型是随机初始化的，范围之后靠训练调整）")

    def setup(self, stage=None):
        # Lightning 在建优化器之前调用这里；从随机初始化开始时就在这一步装量化器
        self._install_quantizers()

    def load_safetensors_checkpoint(self, path):
        """可选：从一份已有的预训练权重出发继续预训练（run_train.py 在给了
        pretrained_safetensors_path 时会调用）。权重文件必须包含解码器。"""
        from safetensors.torch import load_file
        state = {k[len("model."):] if k.startswith("model.") else k: v
                 for k, v in load_file(str(path)).items()}
        if not any(k.startswith("decoder.") for k in state):
            raise ValueError(f"{path} 里没有解码器权重，不能用来继续预训练。")
        self.model.load_state_dict(state, strict=True)  # 先读浮点权重
        print(f"[QAP] 已读取预训练权重（{len(state)} 个张量）：{path}")
        self._install_quantizers()                       # 再装量化器

    # ------------------------------------------------------------------
    # 掩码重建（同作者的 MaskTask）
    # ------------------------------------------------------------------
    def generate_mask(self, batch_size, C, T):
        patch_h, patch_w = self.patch_size
        num_rectangles = (C // patch_h) * (T // patch_w)
        num_to_mask = int(num_rectangles * self.masking_ratio)
        rectangles = [(i, j) for i in torch.arange(0, C, patch_h) for j in torch.arange(0, T, patch_w)]
        selected = torch.randperm(num_rectangles)[:num_to_mask]
        mask = torch.zeros(batch_size, C, T, dtype=torch.bool, device=self.device)
        for idx in selected:
            r, c = rectangles[idx]
            mask[:, r:r + patch_h, c:c + patch_w] = True
        return mask

    def _masked_loss(self, batch):
        X = as_signal_batch(batch)["input"]
        mask = self.generate_mask(X.shape[0], X.shape[1], X.shape[2])
        if self.normalize:
            X = self.normalize_fct(X)
        reconstructed, original = self.model(X, mask)
        masked_loss, _ = self.criterion(reconstructed, original, mask)
        return masked_loss

    def training_step(self, batch, batch_idx):
        loss = self._masked_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False,
                 prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._masked_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_end(self):
        trainer = self.trainer
        if not trainer.is_global_zero:
            return
        if trainer.global_step != int(trainer.estimated_stepping_batches):
            print(f"[QAP] 训练没有跑满（{trainer.global_step} / {int(trainer.estimated_stepping_batches)} 步），"
                  "不保存 final.ckpt")
            return
        out_dir = str(self.hparams.io.base_output_path)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "final.ckpt")
        trainer.save_checkpoint(path)
        print(f"[QAP] 最后一轮的权重已保存：{path}（共训练 {trainer.global_step} 步）")

    def configure_optimizers(self):
        self._install_quantizers()  # 正常情况下 setup 已经装过，这里只是保险
        lr = float(self.hparams.optimizer.lr)
        name = self.hparams.optimizer.optim
        if name == "LAMB":
            optimizer = torch_optim.Lamb(self.model.parameters(), lr=lr)
        elif name == "AdamW":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"预训练只支持 LAMB 或 AdamW，收到 {name}。")

        if self.lr_schedule == "const":
            print(f"[QAP] 学习率：全程固定 {lr:g}")
            return optimizer

        total_steps = int(self.trainer.estimated_stepping_batches)
        max_epochs = int(self.trainer.max_epochs)
        if total_steps < 2 or max_epochs < 1:
            raise RuntimeError(f"总步数（{total_steps}）或轮数（{max_epochs}）不对，无法建立余弦调度。")
        warmup_steps = round(total_steps * self.warmup_epochs / max_epochs)
        warmup_steps = max(1, min(warmup_steps, total_steps // 2))  # 预热最多占一半
        factor = make_warmup_cosine(warmup_steps, total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        print(f"[QAP] 学习率：预热 {warmup_steps} 步升到 {lr:g}，之后余弦衰减，共 {total_steps} 步")
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


def make_warmup_cosine(warmup_steps, total_steps):
    """返回“第 step 步的学习率 = 峰值 x 多少”的函数（step 从 0 数起）。"""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return factor
