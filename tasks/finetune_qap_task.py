"""从我们自己预训练得到的 checkpoint 出发，在 TUAR 上微调。

用同一个任务跑三个组，区别只在量化什么时候打开：

  组          预训练 checkpoint       qap_weight_bits      结果
  浮点参照     浮点预训练的            0                    浮点模型（之后也用来做 PTQ）
  QAT         浮点预训练的            8 或 4               微调从第一步起带量化
  QAP         带量化预训练的          和预训练相同          预训练和微调都带量化

做的事：
  1. 从预训练 checkpoint 里取出编码器的权重（丢掉重建解码器），分类头保持随机初始化。
  2. 按上表决定是否装量化器。量化的位置是全部 28 处权重（编码器 21 + 分类头 7），
     用 models/weight_quant.py，和预训练阶段是同一个量化器。
     QAP 组：编码器的量化尺度沿用预训练学到的值。
     QAT 组：量化尺度由读入的浮点权重重新计算。
  3. 其余训练流程（损失、指标、学习率调度、逐层学习率）完全沿用作者的 FinetuneTask；
     唯一的改动是量化尺度不参与 weight decay（否则会被不断拉向固定值）。

激活量化（qap_act_bits = 8 或 4，models/act_quant.py 的 32 个位置）按同样的规则：
  QAT：读入浮点预训练的权重后装上，全部量化器在训练开始时用训练集校准，之后学习。
  QAP：编码器的量化器沿用预训练学到的范围；分类头的量化器在训练开始时校准。

命令行：task=finetune_qap_task pretrained_safetensors_path=null
        pretrained_checkpoint_path=<预训练 ckpt> +qap_weight_bits=0、8 或 4
"""
import torch

from biofoundation.core.batch import as_signal_batch
from tasks.finetune_task import FinetuneTask


class QAPFinetuneTask(FinetuneTask):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.weight_bits = int(self.hparams.get("qap_weight_bits", 0))
        if self.weight_bits not in (0, 8, 4):
            raise ValueError(f"qap_weight_bits 只支持 0（浮点）、8 或 4，收到 {self.weight_bits}。")
        self.act_bits = int(self.hparams.get("qap_act_bits", 0))
        if self.act_bits not in (0, 8, 4):
            raise ValueError(f"qap_act_bits 只支持 0（浮点）、8 或 4，收到 {self.act_bits}。")
        self.act_percentile = float(self.hparams.get("qap_act_percentile", 99.999))
        self._act_prefix = ""   # 训练开始时要校准的激活量化器："" = 全部，"head_" = 只有分类头
        self._loaded = False

    def _install_act_quant(self):
        from models.act_quant import install_act_quant
        aq = install_act_quant(self.model, self.act_bits)
        expected = 3 + 11 * int(self.hparams.model.num_blocks) + 7
        if len(aq.q) != expected:
            raise RuntimeError(f"应该有 {expected} 个激活量化器，实际 {len(aq.q)} 个。")
        return len(aq.q)

    def on_train_start(self):
        if self.act_bits == 0:
            return
        from models.act_quant import calibrate_in_training

        def run_batch(batch):
            X = as_signal_batch(batch)["input"].to(self.device)
            if getattr(self, "normalize", False):
                X = self.normalize_fct(X)
            mask = self.generate_fake_mask(X.shape[0], X.shape[1], X.shape[2])
            self.model(X, mask)

        aq = self.model.act_quant
        seen, count = calibrate_in_training(
            self, aq, self.act_bits, self.act_percentile, run_batch,
            self.trainer.datamodule.train_dataloader(), n_batches=24, prefix=self._act_prefix)
        which = "分类头的" if self._act_prefix else "全部"
        print(f"[QAP] 激活量化已校准：{which} {count} 个量化器（{self.act_percentile:g}% 分位数，训练集的 {seen} 批）；"
              f"现在打开的激活量化器共 {aq.count_enabled()} / {len(aq.q)} 个")
        if aq.count_enabled() != len(aq.q):
            raise RuntimeError("有激活量化器没有打开。")

    def _install_quantizers(self):
        from models.weight_quant import add_weight_quant
        names = add_weight_quant(self.model, self.weight_bits)  # 默认范围：patch_embed、mamba_blocks、classifier
        expected = 1 + 10 * int(self.hparams.model.num_blocks) + 7
        if len(names) != expected:
            raise RuntimeError(f"应该量化 {expected} 个权重，实际 {len(names)} 个。")
        return len(names)

    def load_pretrained_checkpoint(self, model_ckpt):
        """run_train.py 在给了 pretrained_checkpoint_path 时调用。"""
        if self._loaded:
            raise RuntimeError("预训练权重已经读取过了。")
        # 这是自己训练得到的 checkpoint，里面带有配置对象，需要用完整模式读取
        checkpoint = torch.load(model_ckpt, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        encoder = {k: v for k, v in state.items()
                   if k.startswith("model.") and not k.startswith("model.decoder.")}
        if not encoder:
            raise RuntimeError(f"{model_ckpt} 里没有找到编码器权重。")
        pretrain_was_quantized = any(".parametrizations.weight." in k for k in encoder)
        pretrain_has_act = any(k.startswith("model.act_quant.") for k in encoder)
        pretrain_act_bits = int(checkpoint.get("hyper_parameters", {}).get("qap_act_bits", 0)) if pretrain_has_act else 0
        if pretrain_has_act and self.act_bits != pretrain_act_bits:
            raise ValueError(f"这个预训练 checkpoint 的激活是带 {pretrain_act_bits} bit 量化训练的，"
                             f"微调时请用 +qap_act_bits={pretrain_act_bits}（现在是 {self.act_bits}）。")

        if pretrain_was_quantized:
            pretrain_bits = int(checkpoint.get("hyper_parameters", {}).get("qap_weight_bits", 8))
            if self.weight_bits != pretrain_bits:
                raise ValueError(f"这个预训练 checkpoint 是带 {pretrain_bits} bit 量化训练的，"
                                 f"微调时请用 +qap_weight_bits={pretrain_bits}（现在是 {self.weight_bits}）。")

        n_quant = 0
        if pretrain_was_quantized:
            # QAP：先装量化器，再读权重，这样预训练学到的量化尺度也一起读进来
            n_quant = self._install_quantizers()
        n_act = 0
        if pretrain_has_act:
            # QAP：编码器的激活量化器（范围、开关）沿用预训练的；分类头的在训练开始时校准
            n_act = self._install_act_quant()
            self._act_prefix = "head_"
        missing, unexpected = self.load_state_dict(encoder, strict=False)
        if unexpected:
            raise RuntimeError(f"checkpoint 里有模型用不上的权重：{list(unexpected)[:5]}")
        not_classifier = [k for k in missing
                          if not k.startswith(("model.classifier.", "model.act_quant.q.head_"))]
        if not_classifier:
            raise RuntimeError(f"编码器有权重没读到：{not_classifier[:5]}")
        if not missing:
            raise RuntimeError("分类头的权重不应该出现在预训练 checkpoint 里。")
        if self.weight_bits > 0 and not pretrain_was_quantized:
            # QAT：读完浮点权重后再装量化器，量化尺度由读入的权重计算
            n_quant = self._install_quantizers()

        if self.act_bits > 0 and not pretrain_has_act:
            # QAT：激活量化器全部在训练开始时校准
            n_act = self._install_act_quant()

        for parameter in self.model.parameters():
            parameter.requires_grad = True
        self._loaded = True

        quantized_pretrain = pretrain_was_quantized or pretrain_has_act
        group = "QAP" if quantized_pretrain else ("QAT" if (self.weight_bits > 0 or self.act_bits > 0) else "浮点参照")
        print(f"[QAP] 组别：{group}；权重位宽：{self.weight_bits if self.weight_bits else '浮点'}；"
              f"激活位宽：{self.act_bits if self.act_bits else '浮点'}")
        if n_act:
            print(f"[QAP] 激活量化器：{n_act} 个"
                  + ("（编码器的沿用预训练的范围，分类头的在训练开始时校准）" if pretrain_has_act
                     else "（全部在训练开始时校准）"))
        if n_quant:
            from models.weight_quant import check_weight_quant
            print(f"[QAP] 当前相对量化误差 {check_weight_quant(self.model)[1]:.4f}")
        print(f"[QAP] 读取编码器权重 {len(encoder)} 个；分类头随机初始化（{len(missing)} 个张量）")
        print(f"[QAP] 被量化的权重张量：{n_quant} 个"
              + ("（编码器的量化尺度沿用预训练的值）" if pretrain_was_quantized else ""))
        print(f"[QAP] 预训练 checkpoint：{model_ckpt}（预训练了 {int(checkpoint.get('global_step', -1))} 步）")

    def configure_optimizers(self):
        if not self._loaded:
            raise RuntimeError("还没有读取预训练 checkpoint：请给出 pretrained_checkpoint_path，"
                               "并把 pretrained_safetensors_path 设为 null。")
        result = super().configure_optimizers()
        scale_ids = set()
        if self.weight_bits > 0:
            from models.weight_quant import iter_weight_quantizers
            scale_ids |= {id(q.log_scale) for _, _, q in iter_weight_quantizers(self.model)}
        if self.act_bits > 0:
            scale_ids |= {id(q.log_range) for q in self.model.act_quant.q.values()}
        if scale_ids:
            changed = 0
            for group in result["optimizer"].param_groups:
                if any(id(p) in scale_ids for p in group["params"]):
                    group["weight_decay"] = 0.0
                    changed += 1
            if changed != len(scale_ids):
                raise RuntimeError("有量化尺度没有进入优化器。")
            print(f"[QAP] {changed} 组量化尺度（权重的步长、激活的范围）不参与 weight decay")
        return result
