"""基线这条线上的 QAT：从作者提供的预训练权重出发，在 TUAR 上带权重量化微调。

和 FP32 基线那次训练的区别只有一处：读入作者的权重之后、训练开始之前，给 28 个权重
装上量化器（models/weight_quant.py，和 QAP 那条线用的是同一个）。其余设置完全相同。

  qap_weight_bits=0      不量化，等同于重新训练一次 FP32 基线（可用来核对）
  qap_weight_bits=8 或 4  微调从第一步起带 8 bit / 4 bit 权重量化

这和 QAP 那条线里的 QAT 是同一个定义（浮点预训练 -> 带量化微调），只是预训练权重换成了作者的。

命令行：task=finetune_base_qat_task pretrained_safetensors_path=<作者的 safetensors>
        pretrained_checkpoint_path=null +qap_weight_bits=4
"""
from safetensors.torch import load_file

from tasks.finetune_qap_task import QAPFinetuneTask


class BaseQATFinetuneTask(QAPFinetuneTask):
    def load_safetensors_checkpoint(self, model_ckpt):
        """run_train.py 在给了 pretrained_safetensors_path 时调用。"""
        if self._loaded:
            raise RuntimeError("预训练权重已经读取过了。")
        state = load_file(str(model_ckpt))
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"权重文件里有模型用不上的权重：{list(unexpected)[:5]}")
        not_classifier = [k for k in missing if not k.startswith("model.classifier.")]
        if not_classifier:
            raise RuntimeError(f"编码器有权重没读到：{not_classifier[:5]}")
        if not missing:
            raise RuntimeError("分类头的权重不应该出现在预训练权重文件里。")

        n_quant = 0
        if self.weight_bits > 0:
            # 读完浮点权重后再装量化器，量化步长由读入的权重计算
            n_quant = self._install_quantizers()
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        self._loaded = True

        print(f"[QAP] 组别：基线 QAT（作者的预训练权重）；权重位宽：{self.weight_bits if self.weight_bits else '浮点'}")
        print(f"[QAP] 读取编码器权重 {len(state)} 个；分类头随机初始化（{len(missing)} 个张量）")
        print(f"[QAP] 被量化的权重张量：{n_quant} 个")
        if n_quant:
            from models.weight_quant import check_weight_quant
            print(f"[QAP] 当前相对量化误差 {check_weight_quant(self.model)[1]:.4f}")
        print(f"[QAP] 预训练权重：{model_ckpt}")

    def load_pretrained_checkpoint(self, model_ckpt):
        raise RuntimeError("这个任务只从作者的 safetensors 权重出发；"
                           "从我们自己的预训练 checkpoint 出发请用 task=finetune_qap_task。")
