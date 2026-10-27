# FEMBA-QAP:FEMBA 的量化感知预训练

本仓库在 [BioFoundation](https://github.com/pulp-bio/BioFoundation)(提交 `f7e9437`)的基础上,实现 FEMBA-Tiny 的量化感知预训练(Quantization-Aware Pre-training,QAP),并在相同训练量下与训练后量化(PTQ)和量化感知训练(QAT)进行对比。支持 8 位与 4 位权重量化,以及 8 位与 4 位激活量化。实验数据为按受试者划分的 TUAR 伪迹检测数据集。

BioFoundation 的原有文件均未修改,其 README 保留为 [`README_BioFoundation.md`](README_BioFoundation.md)。新增文件列于第 2 节,也可通过 `git diff --stat f7e9437` 查看。

## 目录

1. [方法概述](#1-方法概述)
2. [目录结构](#2-目录结构)
3. [环境配置](#3-环境配置)
4. [数据准备](#4-数据准备)
5. [使用方法](#5-使用方法)
6. [实验结果](#6-实验结果)
7. [致谢与许可](#7-致谢与许可)

## 1. 方法概述

### 1.1 训练流程

```
无标签 EEG ──► 掩码重建预训练 ──► final.ckpt
               (tasks/pretrain_qap_task.py)
                                      │
                                      ▼
带标签 EEG ──► 分类微调 ─────────► epoch=*.ckpt
               (tasks/finetune_qap_task.py)
                                      │
                                      ▼
               评估 ──────────────► *_metrics.json
               (eval_femba.py)
```

五种对比设置在各阶段的精度如下:

| 设置 | 预训练 | 微调 | 评估 |
|---|---|---|---|
| 浮点基线 | 浮点 | 浮点 | 浮点 |
| PTQ | 浮点 | 浮点 | 评估前直接量化 |
| QAT | 浮点 | 量化 | 量化 |
| QAP | 量化 | 量化 | 量化 |
| 无预训练 | — | 浮点,随机初始化 | 浮点 |

各设置的训练量相同:预训练 50 轮(10,800 步),微调 10 轮,随机种子 42。QAT 与 QAP 的唯一区别在于预训练阶段是否量化。QAP 微调时,编码器沿用预训练阶段学得的量化步长;QAT 的量化步长由浮点权重初始化。重建解码器仅在预训练阶段使用,不量化。

### 1.2 量化方案

#### 权重量化(`models/weight_quant.py`)

采用逐输出通道的对称量化,零点为 0:

$$\hat w = s_c \cdot \mathrm{clip}\left(\mathrm{round}\left(\frac{w}{s_c}\right),\,-q_{\max},\,q_{\max}\right),\qquad q_{\max}=2^{b-1}-1$$

其中 $b$ 为位宽,$s_c$ 为第 $c$ 个输出通道的步长。

- 步长可学习(LSQ),以对数形式存储,梯度按 $1/\sqrt{N q_{\max}}$ 缩放,$N$ 为每个通道的权重数。
- 直通估计(STE)仅作用于取整运算:范围内权重的梯度为 1,被截断权重的梯度为 0。
- 步长初始化可选 `max`(各通道绝对值最大值除以 $q_{\max}$)或 `mse`(在最大值的 30%–100% 区间内搜索量化误差最小的截断点)。
- 量化器通过 `torch.nn.utils.parametrize` 挂载于 `module.weight`,因此 Mamba 内部对权重的所有读取均使用量化值。
- 量化对象共 28 个权重张量:patch 卷积 1 个;编码器中每个 Mamba 的 `in_proj`、`conv1d`、`x_proj`、`dt_proj`、`out_proj`(2 层 × 正反向 2 个 × 5 个 = 20 个);分类头 7 个(2 个线性层及分类头 Mamba 的 5 个)。
- 偏置、位置编码、`A_log`、`D` 及 LayerNorm 保持浮点。

#### 激活量化(`models/act_quant.py`)

采用逐张量量化,共 32 个量化位置。对称量化器:

$$\hat x = s\cdot\mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right),\,-Q,\,Q\right),\qquad Q=2^{b-1}-1,\quad s=\frac{r}{Q}$$

SiLU 与 GELU 的输出存在下界,采用非对称量化器:下界 $l$ 固定为激活函数的最小值,上界可学习。记 $r$ 为上界与下界之差,则 $s=r/(2^b-1)$,零点 $z=\mathrm{round}(-l/s)$:

$$\hat x = s\left(\mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right)+z,\,0,\,2^b-1\right)-z\right)$$

量化位置如下(①–⑤ 为单个 Mamba 内的位置):

| 模块 | 量化位置 | 量化器 |
|---|---|---|
| 嵌入层 | 输入;卷积输出与位置编码(共享步长);二者之和 | 对称 |
| Mamba | ① `in_proj` 输出;② 卷积输出;④ SSM 输出(门控之后);⑤ `out_proj` 输出 | 对称 |
| Mamba | ③ SiLU 输出(SSM 输入) | 非对称,$l=-0.2785$ |
| 编码器层 | 正、反向 Mamba 的 ⑤ 共享步长;正反向相加之后;LayerNorm 之后 | 对称 |
| 分类头 | `fc1` 输出;分类头 Mamba 的 ①–⑤;时间平均之后(沿用 ⑤ 的步长) | 对称 |
| 分类头 | GELU 输出 | 非对称,$l=-0.170$ |

以下部分保持浮点:`x_proj` 的输出、$\Delta$、$B$、$C$、递推隐状态、z 支路的 SiLU 与门控乘法、残差相加、LayerNorm 内部计算、输出 logits。

**顺序校准。** 量化范围的初值由校准数据上的分位数确定,之后在 QAT 与 QAP 中继续学习。输入中的大幅值伪迹经第一层 SSM 后被放大数个数量级(第 0 层 SSM 输出的 99.9% 分位数约为 5250,最大值约为 $4\times10^7$)。若在浮点模型上同时校准所有位置,后续位置的范围将严重失真。因此 `calibrate_sequential` 按前向顺序逐个校准,校准某一位置时,其之前的量化器均已启用。

**实现说明。** `mamba_ssm` 的 Mamba 前向为融合实现,无法获取中间结果。`act_quant.py` 按 `mamba_ssm` 2.2.6 的非融合路径(`mamba_simple.py`)重写了前向并插入量化器,递推部分仍调用 `selective_scan_fn`。量化器全部关闭时直接调用原模块,输出与原模型逐位一致;重写路径的正确性由 `scan_act_quant.py` 校验。

## 2. 目录结构

以下为本仓库在 BioFoundation 之上新增的文件:

```
.
├── models/
│   ├── weight_quant.py             权重量化器
│   └── act_quant.py                激活量化器、量化位置安装、顺序校准
├── tasks/
│   ├── pretrain_qap_task.py        掩码重建预训练(可选权重/激活量化)
│   ├── finetune_qap_task.py        分类微调(浮点基线、QAT、QAP)
│   └── finetune_base_qat_task.py   基于原作者预训练权重的 QAT
├── config/
│   ├── task/
│   │   ├── pretrain_qap_task.yaml
│   │   ├── finetune_qap_task.yaml
│   │   └── finetune_base_qat_task.yaml
│   └── data_module/
│       └── pretrain_tuar_subject.yaml   预训练数据配置(仅读取信号)
├── eval_femba.py                   评估脚本(含 PTQ)
├── scan_act_quant.py               激活量化 PTQ 扫描(分位数选择、敏感度分析)
├── check_final_save.py             final.ckpt 保存逻辑检查
├── make_report.py                  训练日志与评估指标汇总
└── *.sbatch                        Slurm 提交脚本(见 5.6 节)
```

## 3. 环境配置

本仓库的依赖与 BioFoundation 相同,另需安装 `mamba-ssm` 与 `causal-conv1d`。本文实验使用 PyTorch 2.7.1(CUDA 12.6)与 `mamba-ssm` 2.2.6。

```bash
git clone <仓库地址> femba-qap
cd femba-qap
pip install -r requirements.txt
pip install causal-conv1d mamba-ssm==2.2.6 --no-build-isolation
```

`mamba-ssm` 与 `causal-conv1d` 需与 PyTorch 和 CUDA 版本匹配,通常需在带 GPU 的节点上安装。激活量化的前向实现依赖 `mamba-ssm` 2.2.6 的内部结构,使用其他版本时需核对 `mamba_simple.py` 是否一致。

运行前需设置两个环境变量(BioFoundation 的配置文件会读取它们):

```bash
export DATA_PATH=/path/to/data          # 存放 train.h5、val.h5、test.h5 的目录
export CHECKPOINT_DIR=/path/to/run      # 本次运行的输出目录,每次运行应使用不同目录
```

## 4. 数据准备

**数据格式。** 数据需为 BioFoundation 的 `HDF5Loader`(`datasets/hdf5_dataset.py`)可读取的 HDF5 文件,分为 `train.h5`、`val.h5`、`test.h5`。制作方法见 [`make_datasets/README.md`](make_datasets/README.md)。

**本文实验所用数据。** TUAR 伪迹检测数据集,按受试者划分为训练集、验证集、测试集(167 / 19 / 23 名受试者)。每个样本为 22 通道 × 1280 采样点。

**划分原则。**

- 三个子集的受试者互不重叠。
- 预训练仅使用训练集的信号,不使用标签;验证集仅用于监控重建损失。
- 所有模型选择与超参数选择均在验证集上完成;测试集仅对最终配置各评估一次。

**使用其他数据集。**

- 在命令行中通过 `data_module.train.hdf5_file`、`data_module.val.hdf5_file`、`data_module.test.hdf5_file` 指定数据文件。`config/data_module/pretrain_tuar_subject.yaml` 中的路径可由命令行覆盖,无需修改文件。
- 若通道数或序列长度与上述不同,需相应调整 `model.*` 参数与预训练的掩码块大小(默认 2 通道 × 16 采样点)。该情形未经验证。
- 激活量化的校准分位数需重新选择,见 5.5 节。

## 5. 使用方法

以下命令均在仓库根目录执行,示例为 W4(权重 4 位、激活浮点)的 QAP。

### 5.1 量化器自检

```bash
python -m models.weight_quant
python -m models.act_quant
```

前者检查权重量化的取值范围、梯度及保存与加载的一致性;后者检查对称与非对称量化器、截断及校准。均可在 CPU 上运行。`models.weight_quant` 末尾的对比测试依赖本仓库未包含的 `models.qat_w8a32`,运行时会提示跳过,属正常现象。

### 5.2 预训练

```bash
export CHECKPOINT_DIR=$PWD/runs/pretrain_w4
python -u run_train.py \
  +experiment=FEMBA_pretrain task=pretrain_qap_task data_module=pretrain_tuar_subject \
  data_module.train.hdf5_file=$DATA_PATH/train.h5 \
  data_module.val.hdf5_file=$DATA_PATH/val.h5 \
  tag=pretrain_w4 model.embed_dim=35 model.num_blocks=2 \
  gpus=1 trainer.devices=1 trainer.strategy=auto batch_size=256 num_workers=0 \
  trainer.max_epochs=50 \
  +qap_weight_bits=4 +qap_act_bits=0 +qap_lr_schedule=cosine +qap_warmup_epochs=2 \
  final_validate=False final_test=False io.base_output_path=$CHECKPOINT_DIR resume=False
```

| 参数 | 取值 | 说明 |
|---|---|---|
| `+qap_weight_bits` | `0`、`8`、`4` | 编码器权重位宽;`0` 表示浮点预训练 |
| `+qap_act_bits` | `0`、`8`、`4` | 编码器激活位宽;`0` 表示不量化激活 |
| `+qap_act_percentile` | 默认 `99.999` | 激活范围校准所用分位数 |
| `+qap_lr_schedule` | `cosine`、`const` | `cosine`:线性预热后余弦衰减;`const`:恒定学习率 |
| `+qap_warmup_epochs` | 默认 `2` | 预热轮数 |
| `io.base_output_path` | 路径 | `final.ckpt` 的保存目录 |

**输出。** 训练结束时,最后一轮的权重保存为 `$CHECKPOINT_DIR/final.ckpt`。日志中出现 `[QAP] 编码器权重带 4 bit 量化预训练:21 个权重张量` 表示量化已启用。

### 5.3 微调

```bash
export CHECKPOINT_DIR=$PWD/runs/finetune_qap_w4
python -u run_train.py \
  +experiment=FEMBA_finetune task=finetune_qap_task \
  pretrained_safetensors_path=null \
  pretrained_checkpoint_path=$PWD/runs/pretrain_w4/final.ckpt \
  +qap_weight_bits=4 +qap_act_bits=0 \
  tag=finetune_qap_w4 model.embed_dim=35 model.num_classes=2 model.classification_type=bc \
  data_module.train.hdf5_file=$DATA_PATH/train.h5 \
  data_module.val.hdf5_file=$DATA_PATH/val.h5 \
  data_module.test.hdf5_file=$DATA_PATH/test.h5 \
  gpus=1 trainer.devices=1 trainer.strategy=auto \
  batch_size=32 +trainer.accumulate_grad_batches=32 num_workers=0 \
  trainer.max_epochs=10 \
  final_validate=False final_test=False io.base_output_path=$CHECKPOINT_DIR resume=False
```

| 参数 | 说明 |
|---|---|
| `pretrained_checkpoint_path` | 预训练得到的 `final.ckpt`;仅读取编码器,分类头随机初始化 |
| `+qap_weight_bits` | 微调时的权重位宽;若预训练已量化,必须与预训练位宽一致 |
| `+qap_act_bits`、`+qap_act_percentile` | 激活位宽与校准分位数,见 5.5 节 |

按第 1.1 节的表格设置参数,即可得到其他设置:

| 设置 | 预训练 `+qap_weight_bits` | 微调 `+qap_weight_bits` | 微调读取的预训练权重 |
|---|---|---|---|
| 浮点基线 | `0` | `0` | 浮点预训练的 `final.ckpt` |
| QAT | `0` | `4` | 浮点预训练的 `final.ckpt` |
| QAP | `4` | `4` | 量化预训练的 `final.ckpt` |
| 无预训练 | — | — | 使用 `task=finetune_task pretrained_safetensors_path=null pretrained_checkpoint_path=null` |

**输出。** 验证损失最低一轮的权重保存为 `$CHECKPOINT_DIR/checkpoints/<tag>/<时间戳>/epoch=*.ckpt`,可用以下命令查找:

```bash
find $CHECKPOINT_DIR/checkpoints -name "epoch=*.ckpt"
```

### 5.4 评估

```bash
# QAT / QAP / 浮点基线:直接评估微调后的模型(量化设置从 checkpoint 中自动识别)
python eval_femba.py --checkpoint <epoch=*.ckpt> --name qap_w4 --out-dir runs/results --split val

# PTQ:对浮点基线的 checkpoint 直接量化后评估
python eval_femba.py --checkpoint <浮点基线的 epoch=*.ckpt> --name ptq_w4 --out-dir runs/results \
  --split val --ptq-bits 4
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--checkpoint` | — | 待评估的 checkpoint |
| `--name` | — | 输出文件名前缀 |
| `--out-dir` | — | 输出目录 |
| `--split` | `val` | `val` 或 `test` |
| `--data-file` | checkpoint 配置中的路径 | 指定其他 HDF5 文件 |
| `--ptq-bits` | `0` | PTQ 权重位宽(`8` 或 `4`);`0` 表示不做 PTQ |
| `--ptq-init` | `max` | PTQ 步长初始化方式(`max` 或 `mse`) |
| `--ptq-act-bits` | `0` | PTQ 激活位宽(`8` 或 `4`) |
| `--ptq-act-percentile` | `99.999` | PTQ 激活校准分位数 |
| `--calib-windows` | `1024` | 激活校准所用训练窗口数 |

**输出。** `--out-dir` 下生成以下文件:

| 文件 | 内容 |
|---|---|
| `<name>_<split>_metrics.json` | AUROC、AUPR、准确率、平衡准确率、F1、ECE(15 个等宽区间)、交叉熵损失、AUROC 95% 置信区间(按受试者重采样 1000 次) |
| `<name>_<split>_predictions.npz` | 每个窗口的 logits、标签、受试者编号 |
| `<name>_<split>_per_subject.csv` | 每名受试者的窗口数、正样本比例与准确率 |

准确率、平衡准确率与 F1 以概率 0.5 为阈值,"有伪迹"为正类。

### 5.5 激活量化

预训练与微调均添加以下参数,且两个阶段的设置须一致:

| 配置 | 参数 |
|---|---|
| W8A8 | `+qap_weight_bits=8 +qap_act_bits=8 +qap_act_percentile=99.9` |
| W4A8 | `+qap_weight_bits=4 +qap_act_bits=8 +qap_act_percentile=99.9` |
| W4A4 | `+qap_weight_bits=4 +qap_act_bits=4 +qap_act_percentile=95` |

训练开始前,程序取训练集的前若干批数据按前向顺序校准各量化位置,之后量化范围随训练更新。QAP 微调时,编码器的激活量化范围沿用预训练结果,仅分类头重新校准。

**分位数选择。** 上表中的分位数是在 TUAR 验证集上通过 PTQ 选出的。使用其他数据集时,应在一个已微调的浮点模型上重新选择:

```bash
python scan_act_quant.py --checkpoint <浮点基线的 epoch=*.ckpt> --name ref --out-dir runs/results --quick
```

该脚本输出各分位数下的验证集指标,并检验重写的前向与原模型是否一致,以及逐类量化位置的敏感度。去掉 `--quick` 可执行完整扫描。

### 5.6 Slurm 脚本

以下脚本将 5.2–5.5 节的命令封装为 Slurm 作业。脚本中的数据路径、Python 环境路径、输出路径与 GPU 型号需根据实际环境修改。

| 脚本 | 功能 | 参数 |
|---|---|---|
| `pretrain_tuar_qap.sbatch` | 预训练 | `<权重位宽> <轮数> <cosine\|const> [full\|smoke] [激活位宽] [分位数]` |
| `finetune_tuar_qap.sbatch` | 微调 | `<名称> <权重位宽> <预训练 ckpt\|none> [full\|smoke] [激活位宽] [分位数]` |
| `chain_qap.sbatch` | 预训练后接续微调 | `<权重位宽> [激活位宽] [分位数]`;权重位宽为 `0` 时依次微调浮点基线、W4 与 W8 的 QAT |
| `qat_act_chain.sbatch` | 带激活量化的 QAT(W8A8、W4A4、W4A8) | 无;需已有浮点预训练的 `final.ckpt` |
| `finetune_tuar_base_qat.sbatch` | 基于原作者预训练权重的 QAT | `<权重位宽> [smoke]` |
| `eval_tuar.sbatch` | 评估单个 checkpoint | `<ckpt> <名称> <val\|test\|both> [PTQ 位宽] [max\|mse]` |
| `eval_all_qap.sbatch` | 评估全部已完成的微调,并对浮点基线执行 PTQ | `[val\|test\|both]` |
| `scan_act_quant.sbatch` | 激活量化 PTQ 扫描 | `[full\|quick] [ckpt] [名称] [权重位宽]` |
| `smoke_act_train.sbatch` | 带激活量化训练的小规模测试 | `[分位数]` |

示例:

```bash
sbatch chain_qap.sbatch 0          # 浮点预训练,随后微调浮点基线、QAT W4、QAT W8
sbatch chain_qap.sbatch 4          # W4 QAP
sbatch chain_qap.sbatch 4 4 95     # W4A4 QAP
sbatch eval_all_qap.sbatch both    # 在验证集与测试集上评估
```

参数 `smoke` 表示小规模测试:每轮只运行少量批次,用于确认流程可以运行,结果无参考意义。

### 5.7 注意事项

- **使用 `final.ckpt` 而非 `last.ckpt`。** PyTorch Lightning 生成的 `last.ckpt` 仅在验证损失创新低时更新,其内容等同于最佳轮,而非最后一轮。为保证各设置的预训练量一致,微调一律读取 `final.ckpt`。
- **位宽一致性。** 若预训练已量化,微调的 `+qap_weight_bits` 与 `+qap_act_bits` 必须与预训练相同,否则程序报错。
- **小规模测试。** 正式运行前,建议在预训练命令末尾添加 `+trainer.limit_train_batches=6 +trainer.limit_val_batches=2 trainer.log_every_n_steps=1`,在微调命令末尾添加 `+trainer.limit_train_batches=64 +trainer.limit_val_batches=2` 并设置 `trainer.max_epochs=1`。
- **量化步长不参与权重衰减。** 微调时,权重的量化步长与激活的量化范围均从 weight decay 中排除。

## 6. 实验结果

**实验设置。** FEMBA-Tiny(`embed_dim=35`,2 个双向 Mamba 块),TUAR 数据集按受试者划分(训练 / 验证 / 测试:167 / 19 / 23 名受试者),随机种子 42。表中数值格式为"验证集 / 测试集"。

**AUROC**(浮点基线:0.8940 / 0.8721;无预训练:0.9002 / 0.8906)

| 配置 | PTQ | QAT | QAP |
|---|---|---|---|
| W8A32 | 0.8940 / 0.8719 | 0.8945 / 0.8827 | 0.8969 / 0.8920 |
| W4A32 | 0.8925 / 0.8691 | 0.8947 / 0.8819 | 0.8978 / 0.8817 |
| W8A8 | 0.8380 / 0.8712 | 0.8732 / 0.8950 | 0.8982 / 0.8776 |
| W4A8 | 0.8398 / 0.8721 | 0.8740 / 0.8975 | 0.8578 / 0.8586 |
| W4A4 | 0.7998 / 0.8540 | 0.8348 / 0.8453 | 0.8487 / 0.8501 |

**平衡准确率**(浮点基线:0.799 / 0.793)

| 配置 | PTQ | QAT | QAP |
|---|---|---|---|
| W8A8 | 0.668 / 0.714 | 0.805 / 0.819 | 0.812 / 0.786 |
| W4A8 | 0.673 / 0.719 | 0.804 / 0.816 | 0.794 / 0.791 |
| W4A4 | 0.573 / 0.598 | 0.800 / 0.803 | 0.804 / 0.801 |

**期望校准误差 ECE**(浮点基线:0.049 / 0.048,越低越好)

| 配置 | PTQ | QAT | QAP |
|---|---|---|---|
| W8A8 | 0.170 / 0.119 | 0.092 / 0.108 | 0.071 / 0.058 |
| W4A8 | 0.164 / 0.113 | 0.097 / 0.113 | 0.040 / 0.044 |
| W4A4 | 0.122 / 0.149 | 0.173 / 0.175 | 0.191 / 0.187 |

**主要结论。**

1. 仅量化权重时,PTQ 至 4 位相对同一浮点模型的 AUROC 损失不超过 0.003,未观察到 PTQ 与 QAP 的交叉点。
2. 引入激活量化后,PTQ 的平衡准确率与 ECE 在验证集和测试集上均明显变差;QAT 与 QAP 将平衡准确率恢复至 0.786–0.819。PTQ 的 AUROC 仅在验证集上明显下降,在测试集上变化不明显。
3. 逐位置的 PTQ 敏感度分析表明,SSM 输出(位置 ④)对激活量化最为敏感:单独量化该位置时,验证集 AUROC 在 8 位与 4 位下分别下降 0.063 与 0.080。
4. QAP 与 QAT 之间未观察到稳定差异:W4A8 下 QAP 在两个数据集上均低于 QAT,W4A4 下略高于 QAT,W8A8 下两个数据集的结论相反。

**局限。** 以上结果均来自单一随机种子;验证集与测试集的受试者数较少,各设置的 AUROC 95% 置信区间相互重叠。带激活量化的微调在第 10 轮时验证损失仍在下降,尚未收敛。PTQ 仅采用了分位数校准。

## 7. 致谢与许可

本仓库基于苏黎世联邦理工学院(ETH Zurich)的 [BioFoundation](https://github.com/pulp-bio/BioFoundation) 代码库,其源代码采用 Apache-2.0 许可,详见 [`LICENSE`](LICENSE)。本仓库新增文件采用相同许可。

FEMBA 模型请引用:

```bibtex
@inproceedings{tegon2025femba,
  title={{FEMBA}: Efficient and Scalable {EEG} Analysis with a Bidirectional Mamba Foundation Model},
  author={Tegon, Anna and Ingolfsson, Thorir Mar and Wang, Xiaying and Benini, Luca and Li, Yawei},
  booktitle={2025 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  year={2025},
  pages={1--7},
  doi={10.1109/EMBC58623.2025.11252697}
}
```
