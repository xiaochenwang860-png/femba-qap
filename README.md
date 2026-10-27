# FEMBA 量化感知预训练(QAP):W4A32 / W4A8 / W4A4

NTU EE6008 项目 *QAP vs PTQ Bit-Width Crossover* 的 4 bit 代码。在 FEMBA-Tiny 上做权重和激活的 4 bit 量化,在 TUAR(按受试者划分)上对比四种做法:

| 做法 | 含义 |
|---|---|
| PTQ | 拿训练好的浮点模型直接量化,不再训练 |
| QAT | 浮点预训练 → 带量化微调 |
| QAP | **带量化的预训练** → 带量化微调(本项目要检验的) |
| 浮点参照 / 不预训练 | 对照组 |

本仓库 = 作者的 [BioFoundation](https://github.com/pulp-bio/BioFoundation)(固定在提交 `f7e9437`,保留了原有提交历史,原 README 改名为 `README_BioFoundation.md`)+ 我们新增的文件,新增的文件都放在一个提交里:`git show --stat HEAD` 可以看到我们加了什么。**作者的文件一个都没改**:激活量化是在运行时替换模型实例的 `forward` 实现的。

想直接上手:跳到[第 4 节](#4-怎么用),按 4.3 一步步跑一遍 W4 的 QAP。

## 1. 结果(单个随机种子 42)

AUROC(验证集 / 测试集),浮点参照 0.8940 / 0.8721:

| 位宽 | PTQ | QAT | QAP |
|---|---|---|---|
| W8A32(只量化权重) | 0.8940 / 0.8719 | 0.8945 / 0.8827 | 0.8969 / 0.8920 |
| W4A32(只量化权重) | 0.8925 / 0.8691 | 0.8947 / 0.8819 | 0.8978 / 0.8817 |
| W8A8 | 0.8380 / 0.8712 | 0.8732 / 0.8950 | 0.8982 / 0.8776 |
| W4A8 | 0.8398 / 0.8721 | 0.8740 / 0.8975 | 0.8578 / 0.8586 |
| W4A4 | 0.7998 / 0.8540 | 0.8348 / 0.8453 | 0.8487 / 0.8501 |

平衡准确率(阈值 0.5),浮点参照 0.799 / 0.793:

| 位宽 | PTQ | QAT | QAP |
|---|---|---|---|
| W8A8 | 0.668 / 0.714 | 0.805 / 0.819 | 0.812 / 0.786 |
| W4A8 | 0.673 / 0.719 | 0.804 / 0.816 | 0.794 / 0.791 |
| W4A4 | 0.573 / 0.598 | 0.800 / 0.803 | 0.804 / 0.801 |

对照:浮点模型不预训练直接微调,AUROC 0.9002 / 0.8906。读结果时请注意:

- 只量化权重时,PTQ 到 4 bit 都几乎无损(相对同一个浮点模型最多掉 0.003),没有 QAT/QAP 需要补回的东西,没有出现交叉点。
- 加上激活量化后,PTQ 的**平衡准确率和校准(ECE)**在验证集、测试集上都明显变差,QAT、QAP 把平衡准确率拉回 0.80 左右。PTQ 的 AUROC 只在验证集上明显下降,测试集上没有,不能当作结论。
- QAP 相对 QAT **没有稳定优势**:W4A8 上 QAP 在两个集合上都低于 QAT,W4A4 上略高,W8A8 在验证集和测试集上方向相反。
- 限制:单种子;验证集 19 人、测试集 23 人,AUROC 的 95% 区间互相重叠;带激活量化的微调跑到第 10 轮验证损失还在下降,没有收敛;PTQ 只用了分位数校准一种方法;我们的小规模预训练(只用 TUAR)本身没有比不预训练更好,这会削弱 QAP 的可测效果。

## 2. 量化方案

### 权重(`models/weight_quant.py`)

每个输出通道一个步长 $s_c$,对称,零点为 0:

$$\hat w = s_c \cdot \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{w}{s_c}\right),\,-q_{\max},\,q_{\max}\right),\qquad q_{\max}=2^{b-1}-1$$

4 bit 时 $q_{\max}=7$,8 bit 时为 127。要点:

- 步长可学习(LSQ),以对数形式存储;STE 只作用在取整上,先截断再取整。范围内的权重梯度为 1,被截断的权重梯度为 0,它们的梯度改作用到步长上;步长梯度按 $1/\sqrt{N q_{\max}}$ 缩放($N$ 为每个通道的权重个数)。
- 步长初始化 `max`(每通道最大值 / $q_{\max}$,作者默认做法)或 `mse`(在最大值的 30%–100% 之间找量化误差最小的截断点)。
- 用 `torch.nn.utils.parametrize` 装在 `module.weight` 上,所以 Mamba 内部读 `weight` 的地方拿到的都是量化后的值。
- 量化 28 个权重张量:patch 卷积 1 个 + 每个 Mamba 的 `in_proj`、`conv1d`、`x_proj`、`dt_proj`、`out_proj`(编码器 2 层 × 正反向 2 个 Mamba × 5 个 = 20)+ 分类头 7 个(两个线性层 + 分类头 Mamba 的 5 个)。偏置、位置编码、`A_log`、`D`、LayerNorm 和重建解码器保持浮点。
- 逐通道的原因:深度卷积 `conv1d` 各通道的最大值相差 4–21 倍。我们没有做逐通道与逐张量的对照实验。

### 激活(`models/act_quant.py`)

整个张量一个步长,共 32 个量化位置(embedding 3 个 + 编码器每层 11 个 × 2 层 + 分类头 7 个)。对称量化器:

$$\hat x = s\cdot\mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x}{s}\right),\,-Q,\,Q\right),\qquad Q=2^{b-1}-1,\; s=\frac{r}{Q}$$

SiLU、GELU 输出有下界,用非对称量化器:下界 $l$ 固定为函数的最小值($-0.2785$ 和 $-0.170$),上界可学习,$s=r/(2^b-1)$,$r$ 为上界减下界,零点 $z=\mathrm{round}(-l/s)$:

$$\hat x = s\left(\mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x}{s}\right)+z,\,0,\,2^b-1\right)-z\right)$$

量化位置(以一个 Mamba 为例):① `in_proj` 输出 → ② 卷积输出 → ③ SiLU 输出(非对称,SSM 的输入)→ ④ SSM 输出(门控相乘之后)→ ⑤ `out_proj` 输出。保持浮点:`x_proj` 的输出、$\Delta, B, C$、递推的隐状态、z 路的 SiLU 和门控乘法、残差相加、LayerNorm 的计算、最后的两个得分。

**必须按顺序校准。** 输入里的大幅值伪迹经第一层 SSM 后会被放大几个数量级(第 0 层 SSM 输出 99.9% 分位约 5250,最大值约 $4\times10^7$)。在浮点模型上一次性校准所有位置会让后面位置的范围失真,PTQ 的结果大幅变差;`calibrate_sequential` 按前向顺序逐个位置校准,校准某个位置时前面的量化器已经打开,问题随之消失。

PTQ 的分位数在**验证集**上选定:8 bit 激活用 99.9%,4 bit 激活用 95%。敏感度扫描显示最脆弱的位置是 ④ SSM 输出(验证集 AUROC:8 bit 掉 0.063,4 bit 掉 0.080),这是下一步要重点处理的地方。

`mamba_ssm` 的 Mamba 前向是封装好的,拿不到中间结果,所以 `act_quant.py` 按它的非融合路径(mamba_ssm 2.2.6 的 `mamba_simple.py`)把前向重写了一遍,递推仍然调用它的 `selective_scan_fn`。量化全关时直接调用原模块,结果与原模型逐位相同;重写的路径本身由 `scan_act_quant.py` 检查(量化全关、强制走重写路径,与原模型比较)。

## 3. 文件说明

| 文件 | 作用 |
|---|---|
| `models/weight_quant.py` | 权重量化器,PTQ/QAT/QAP 共用;带 CPU 自检 |
| `models/act_quant.py` | 激活量化器、32 个位置的安装、顺序校准;带 CPU 自检 |
| `tasks/pretrain_qap_task.py` | 掩码重建预训练,可选权重/激活量化;余弦学习率;结束时另存 `final.ckpt` |
| `tasks/finetune_qap_task.py` | 从我们自己的预训练出发微调;同一个任务跑浮点参照、QAT、QAP |
| `tasks/finetune_base_qat_task.py` | 从作者的预训练权重出发做 QAT(基线线) |
| `config/task/*.yaml`、`config/data_module/pretrain_tuar_subject.yaml` | Hydra 配置(任务入口;预训练只读信号不读标签) |
| `pretrain_tuar_qap.sbatch` | 提交预训练 |
| `finetune_tuar_qap.sbatch` | 提交微调(`none` = 不预训练) |
| `chain_qap.sbatch` | 一个任务里连着跑"50 轮预训练 + 对应微调" |
| `qat_act_chain.sbatch` | 从浮点预训练出发,依次做 W8A8、W4A4、W4A8 的 QAT |
| `finetune_tuar_base_qat.sbatch` | 作者权重 + 权重量化微调 |
| `eval_femba.py`、`eval_tuar.sbatch`、`eval_all_qap.sbatch` | 评估:AUROC、AUPR、准确率、平衡准确率、F1、ECE、损失、按受试者重采样的 AUROC 95% 区间;`--ptq-bits`、`--ptq-act-bits` 做 PTQ |
| `scan_act_quant.py/.sbatch` | 激活 PTQ 扫描:检查重写的前向、选分位数、逐类位置看敏感度 |
| `smoke_act_train.sbatch` | 带激活量化训练的冒烟测试(每步只跑几批) |
| `check_final_save.py` | 检查"训练结束保存 `final.ckpt`"在你的环境里正常 |
| `make_report.py` | 把日志和指标汇总成一份 `report.md` |

## 4. 怎么用

### 4.1 QAP 的流程

QAP(quantization-aware pre-training)和普通做法的区别只有一点:**预训练阶段就带着量化**。整条流程分三步,每一步对应一个文件:

```
① 预训练   在无标签信号上做"遮住一部分再重建",编码器权重带 4 bit 量化
           tasks/pretrain_qap_task.py   →  输出 <输出目录>/final.ckpt
                      │
                      ▼
② 微调     读入 final.ckpt 里的编码器(量化步长也一起读入),分类头随机初始化,
           带 4 bit 量化训练分类任务
           tasks/finetune_qap_task.py   →  输出 .../checkpoints/.../epoch=*.ckpt
                      │
                      ▼
③ 评估     eval_femba.py                →  输出 <名字>_<val|test>_metrics.json
```

要比较的几种做法,区别只在每一步的量化开关 `+qap_weight_bits`(0 = 浮点,4 = 权重 4 bit):

| 做法 | ① 预训练 | ② 微调 | ③ 评估 |
|---|---|---|---|
| **QAP** | `+qap_weight_bits=4` | `+qap_weight_bits=4`,读 QAP 的 `final.ckpt` | 直接评估 |
| QAT | `+qap_weight_bits=0`(浮点) | `+qap_weight_bits=4`,读浮点预训练的 `final.ckpt` | 直接评估 |
| 浮点参照 | 同上(浮点预训练) | `+qap_weight_bits=0` | 直接评估 |
| PTQ | 不用另外训练 | 用浮点参照微调出来的 checkpoint | 加 `--ptq-bits 4` |
| 不预训练 | 无 | `task=finetune_task pretrained_safetensors_path=null pretrained_checkpoint_path=null` | 直接评估 |

微调的位宽必须和它读入的预训练一致(QAP 预训练用 4 bit,微调也得是 4 bit),不一致代码会报错。

### 4.2 准备

- **环境**:我们用 PyTorch 2.7.1 + cu126、`mamba_ssm` 2.2.6、`causal-conv1d`,其余依赖见作者的 `requirements.txt`。激活量化重写了 Mamba 的前向,对照的是 `mamba_ssm` 2.2.6 的 `mamba_simple.py`,换版本要核对。需要 GPU。
- **代码**:`git clone <本仓库地址>`,然后 `cd` 进仓库根目录。下面所有命令都在根目录运行。
- **数据**:HDF5 文件 `train.h5`、`val.h5`、`test.h5`,按受试者划分,训练集、验证集、测试集的受试者互不重叠。我们用的是 TUAR(167 / 19 / 23 个受试者),输入是 22 通道 × 1280 个采样点。预训练只读信号不读标签,验证集只用来看重建损失,测试集不参与预训练。
- **两个环境变量**(作者的配置要求):

```bash
export DATA_PATH=<放 h5 文件的目录>
export CHECKPOINT_DIR=<这次运行的输出目录>      # 每次运行换一个目录,见下面的例子
```

### 4.3 跑一遍 W4 的 QAP(不用 Slurm)

**先自检**(CPU 即可,几秒钟):

```bash
python -m models.weight_quant     # 权重量化器:取值只落在允许的等级、梯度正确、存取一致
python -m models.act_quant        # 激活量化器:对称/非对称、截断、校准
```

`models.weight_quant` 最后会试着和我们早期的 `models.qat_w8a32` 对比,本仓库没有那个文件,会打印"跳过",正常。

**① 预训练**(50 轮,在 L40 上约半小时):

```bash
export CHECKPOINT_DIR=$PWD/runs/pretrain_w4
python -u run_train.py \
  +experiment=FEMBA_pretrain task=pretrain_qap_task data_module=pretrain_tuar_subject \
  data_module.train.hdf5_file=$DATA_PATH/train.h5 data_module.val.hdf5_file=$DATA_PATH/val.h5 \
  tag=pretrain_w4 model.embed_dim=35 model.num_blocks=2 \
  gpus=1 trainer.devices=1 trainer.strategy=auto batch_size=256 num_workers=0 \
  trainer.max_epochs=50 \
  +qap_weight_bits=4 +qap_act_bits=0 +qap_lr_schedule=cosine +qap_warmup_epochs=2 \
  final_validate=False final_test=False io.base_output_path=$CHECKPOINT_DIR resume=False
```

跑完后 `runs/pretrain_w4/final.ckpt` 就是最后一轮的权重。屏幕上会打印 `[QAP] 编码器权重带 4 bit 量化预训练:21 个权重张量`,看到这行说明量化确实开了。

**② 微调**(10 轮,在 L40 上约 20 分钟):

```bash
export CHECKPOINT_DIR=$PWD/runs/finetune_qap_w4
python -u run_train.py \
  +experiment=FEMBA_finetune task=finetune_qap_task \
  pretrained_safetensors_path=null pretrained_checkpoint_path=$PWD/runs/pretrain_w4/final.ckpt \
  +qap_weight_bits=4 +qap_act_bits=0 \
  tag=finetune_qap_w4 model.embed_dim=35 model.num_classes=2 model.classification_type=bc \
  data_module.train.hdf5_file=$DATA_PATH/train.h5 \
  data_module.val.hdf5_file=$DATA_PATH/val.h5 \
  data_module.test.hdf5_file=$DATA_PATH/test.h5 \
  gpus=1 trainer.devices=1 trainer.strategy=auto batch_size=32 +trainer.accumulate_grad_batches=32 \
  num_workers=0 trainer.max_epochs=10 \
  final_validate=False final_test=False io.base_output_path=$CHECKPOINT_DIR resume=False
```

屏幕上会打印 `[QAP] 组别:QAP;权重位宽:4`。验证损失最低的那一轮存成 `runs/finetune_qap_w4/checkpoints/.../epoch=*.ckpt`,找它:

```bash
find runs/finetune_qap_w4/checkpoints -name "epoch=*.ckpt"
```

**③ 评估**(默认评估验证集;测试集要显式写 `--split test`):

```bash
python eval_femba.py --checkpoint <上一步找到的 epoch=*.ckpt> --name qap_w4 --out-dir runs/results
python eval_femba.py --checkpoint <同一个 ckpt> --name qap_w4 --out-dir runs/results --split test
```

在 `runs/results/` 下得到 `qap_w4_val_metrics.json`(AUROC、AUPR、准确率、平衡准确率、F1、ECE、损失、按受试者重采样的 AUROC 95% 区间)、`..._predictions.npz`(每个窗口的原始输出)、`..._per_subject.csv`。约定:所有比较和选择都在验证集上做,测试集只给已经确定的配置各用一次。

**对照组**:把上面的 `+qap_weight_bits` 按 4.1 的表改一改即可。PTQ 不用训练,对浮点参照的 checkpoint 直接量化再评估:

```bash
python eval_femba.py --checkpoint <浮点参照的 epoch=*.ckpt> --name ptq_w4 --out-dir runs/results --ptq-bits 4
```

**第一次跑,先试跑**:预训练命令末尾加 `+trainer.limit_train_batches=6 +trainer.limit_val_batches=2 trainer.log_every_n_steps=1`,微调命令末尾加 `+trainer.limit_train_batches=64 +trainer.limit_val_batches=2` 并把 `trainer.max_epochs` 改成 1,几分钟确认整条流程能跑通,再跑正式的。

### 4.4 再加上激活量化(W4A8、W4A4)

预训练和微调的命令都加(或改)这两个参数,两步要一致:

| 目标 | 参数 |
|---|---|
| W4A8 | `+qap_weight_bits=4 +qap_act_bits=8 +qap_act_percentile=99.9` |
| W4A4 | `+qap_weight_bits=4 +qap_act_bits=4 +qap_act_percentile=95` |

`qap_act_percentile` 是校准激活范围时取的分位数。上面两个值是我们在自己数据的验证集上用 PTQ 选出来的(8 bit 激活 99.9,4 bit 激活 95),**换了数据集要重选**:拿一个微调好的浮点 checkpoint 跑 `python scan_act_quant.py --checkpoint <ckpt> --name ref --out-dir runs/results --quick`,看哪个分位数在验证集上最好。这个脚本还会检查重写的前向和原模型是否一致,以及哪一类位置对量化最敏感。

激活量化时,预训练和微调开始前会自动校准:取训练集前几批数据,按前向顺序逐个位置定范围并打开量化器,之后范围随训练一起学习。

### 4.5 用 Slurm 一键跑(我们集群上用的)

`*.sbatch` 是上面几条命令的 Slurm 包装,里面的数据路径、环境路径、输出路径和卡型号是我们集群的示例,按你自己的环境改。

```bash
sbatch chain_qap.sbatch 0           # 浮点预训练 → fp_ref、qat_w4、qat_w8 三个微调
sbatch chain_qap.sbatch 4           # W4 QAP:权重 4 bit 预训练 → 微调
sbatch chain_qap.sbatch 4 8 99.9    # W4A8 QAP(第 3 个参数是激活校准的分位数)
sbatch chain_qap.sbatch 4 4 95      # W4A4 QAP
sbatch qat_act_chain.sbatch         # W8A8、W4A4、W4A8 的 QAT(需要先有浮点预训练的 final.ckpt)
sbatch eval_all_qap.sbatch both     # 评估所有已经微调完的模型(验证集和测试集)
```

也可以分开提交:

```bash
sbatch pretrain_tuar_qap.sbatch 4 50 cosine                       # 参数:权重位宽 轮数 学习率调度 [full|smoke] [激活位宽] [分位数]
sbatch finetune_tuar_qap.sbatch qap_w4 4 <预训练目录>/final.ckpt   # 参数:名字 权重位宽 预训练ckpt [full|smoke] [激活位宽] [分位数]
sbatch finetune_tuar_qap.sbatch scratch 0 none                    # 不预训练的对照
```

`smoke` 参数就是上面的试跑。

### 4.6 换成自己的数据集

1. 数据要整理成作者的 `HDF5Loader`(`datasets/hdf5_dataset.py`)能读的 HDF5 格式,制作方法见作者的 `make_datasets/` 和 `README_BioFoundation.md`。
2. 命令里的 `data_module.*.hdf5_file=...` 指向你的文件。预训练用的 `config/data_module/pretrain_tuar_subject.yaml` 里写的是我们的文件路径,命令行覆盖即可,不用改文件。
3. **输入形状**:我们是 22 通道 × 1280 点,预训练遮挡的小块是 2 通道 × 16 点。你的数据通道数或长度不同的话,要相应改 `model.*` 和遮挡设置,这部分我们没试过。
4. 受试者划分要保证训练、验证、测试的受试者互不重叠,否则指标会偏高。
5. 激活量化的分位数按 4.4 重新选。
6. 量化代码本身不依赖数据,依赖的是 FEMBA 的模型结构(`patch_embed`、双向相加的 `mamba_blocks`、`classifier`)。

### 4.7 在自己的代码里用量化器

```python
from models.weight_quant import add_weight_quant, check_weight_quant
from models.act_quant import install_act_quant, calibrate_sequential

add_weight_quant(model, bits=4)          # 读入权重之后、建优化器之前调用
check_weight_quant(model)                # 确认取值只落在 ±7 的整数格上

aq = install_act_quant(model, bits=4)    # 装上 32 个量化位置,此时全部关闭,输出与原模型相同
# 之后用 calibrate_sequential(aq, run, bits, percentile) 顺序校准并打开
```

## 5. 实验设计和约定

- 所有组除了量化设置,训练量相同:预训练 50 轮(10800 步,批大小 256,前 2 轮预热后余弦衰减,遮挡 60% 的小块,每块 2 通道 × 16 点);微调 10 轮(批大小 32,梯度累积 32,按验证损失保留最佳轮);随机种子 42。QAT 与 QAP 的唯一区别是预训练阶段是否量化。
- 微调一律从 `final.ckpt`(最后一轮)出发。**不要用 `last.ckpt`**:训练程序自带的 `last.ckpt` 只在验证损失创新低时才更新,内容等于最佳轮,不是最后一轮。我们第一次预训练因此三组轮数不一致,全部重跑过。
- QAP 微调时,编码器的量化步长沿用预训练学到的值;QAT 的步长由读入的浮点权重重新计算。量化步长不参与 weight decay。
- 评估脚本的指标全部用 numpy 实现(定义与 scikit-learn 相同,已对照验证),因为我们的环境里没有 scikit-learn。

## 6. 还没做的

三值权重;有效比特数和推理开销;第二个下游任务;交叉点图;多个随机种子或对逐窗口预测做配对自助法;把带激活量化的微调训练到收敛;对 SSM 输出位置 ④ 单独处理(更低的分位数,或旋转类方法)。

## 7. 许可

作者的 BioFoundation 代码按 Apache-2.0 许可(见 `LICENSE`),我们新增的文件放在同一仓库里,同样适用。
