"""检查"预训练结束时保存最后一轮权重"这一步在服务器的环境里是否正常。

用一个极小的模型在 CPU 上训练 3 轮（几秒钟），训练结束时执行的保存代码就是预训练任务里的
同一段（QAPPretrainTask.on_train_end）。然后读回 final.ckpt，确认它记录的训练步数等于总步数。

在 BioFoundation 目录下、登录节点上直接运行，不需要 GPU：
  python check_final_save.py
"""
import os
import tempfile

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from tasks.pretrain_qap_task import QAPPretrainTask


class Tiny(pl.LightningModule):
    on_train_end = QAPPretrainTask.on_train_end  # 和预训练任务用同一段保存代码

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.layer = torch.nn.Linear(4, 1)

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def main():
    out_dir = tempfile.mkdtemp(prefix="check_final_save_")
    model = Tiny(OmegaConf.create({"io": {"base_output_path": out_dir}}))
    loader = DataLoader(TensorDataset(torch.randn(40, 4)), batch_size=4)  # 每轮 10 步
    trainer = pl.Trainer(max_epochs=3, accelerator="cpu", devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.fit(model, loader)

    path = os.path.join(out_dir, "final.ckpt")
    if not os.path.isfile(path):
        raise SystemExit("检查失败：训练结束后没有生成 final.ckpt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    steps = int(checkpoint["global_step"])
    if steps != 30:
        raise SystemExit(f"检查失败：final.ckpt 记录的是 {steps} 步，应为 30 步")
    if "layer.weight" not in checkpoint["state_dict"]:
        raise SystemExit("检查失败：final.ckpt 里没有模型权重")
    print(f"检查通过：final.ckpt 保存的是最后一轮的权重（{steps} / 30 步）")


if __name__ == "__main__":
    main()
