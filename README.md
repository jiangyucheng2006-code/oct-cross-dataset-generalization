# OCT Cross-Dataset Generalization

Basic CNN、Dynamic CNN 与 noise-aware training 的 OCT 四分类实验。
只用 **OCT2017** 训练和选择 checkpoint，在内部测试集、NEH 和 OCT-C8 上分别评价。

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiangyucheng2006-code/oct-cross-dataset-generalization/blob/main/notebooks/OCT_Colab.ipynb)

这是项目的独立论文实现与复现实验代码，不是论文作者的官方仓库。
当前默认 **每种方法运行一次**，固定 seed 42 便于复现。

## 同学从这里开始

1. 点击 **Open in Colab**，选择 `Runtime → Change runtime type → GPU`。
2. 将小组共享的清洗 ZIP 放入自己的 Google Drive，或在 My Drive 添加共享文件夹快捷方式。
3. 按 notebook 顺序执行，修改其中的 `OCT2017_ZIP` 路径。先保留 `MODE = "smoke"`、`VARIANTS = ["basic"]`，确认环境能跑。
4. 试跑成功后，重新使用独立的 run 名称，将 `MODE` 改为 `"full"`，填好 NEH、OCT-C8 的 ZIP 路径，并预先选择要运行的方法。
5. 完整训练后，显式打开最终测试开关，运行评价。训练曲线、checkpoint 和结果会保存到自己的 Drive。

**smoke 只跑两轮、每类最多 64 张训练图和 64 张验证图，不是研究结果，也不会访问测试图像。**
完整模式沿用团队现有划分，不重新随机划分。不要用 smoke 的 checkpoint 做最终评价。
Colab 的 GPU 配额与会话时长有限，完整加噪训练可能需要跨会话恢复。

## 四组主实验

| `variant` | 网络 | 训练方式 |
|---|---|---|
| `basic` | Basic CNN | 原图分类损失 |
| `dynamic` | K=4 Dynamic CNN | 原图分类损失 |
| `basic_noise` | Basic CNN | 原图 + 4 个独立加噪视图 |
| `na_dynamic` | K=4 Dynamic CNN | 原图 + 4 个独立加噪视图，每个视图重新计算 routing |

四组使用相同训练数据、增强、优化器、checkpoint 选择和停止条件。
噪声损失为 `CE(clean) + 0.12 × mean(CE(noisy_1), ..., CE(noisy_4))`；
强度为 `sigma = 0.06`，在 `[0,1]` 强度域加噪后逐图标准化。
验证和测试只做一次正常前向计算，不加入噪声、不做目标域适配。

之前的 `dynamic_stats`（router 增加标准差描述）和 `dynamic_balance`
（router 使用 KL 平衡正则）代码保存在 [`legacy/dynamic_20261107`](legacy/dynamic_20261107)。
旧批次 label smoothing 为 0.0，新四组为 0.1；两批结果应分开报告。

## 数据协议

| 数据 | 用途 | 清洗后数量 / 类别 |
|---|---|---|
| OCT2017 train | 更新模型参数 | 60,663；CNV / DME / DRUSEN / NORMAL |
| OCT2017 val | 选择 checkpoint | 20,246；四类 |
| OCT2017 test | 内部测试 | 20,240；四类 |
| NEH | 外部测试 | 16,663；CNV / DRUSEN / NORMAL |
| OCT-C8 | 外部测试 | 共同四类 1,546；原清洗包共有八类 |

NEH 排除 DME logit 后，在其余三类重新计算 softmax，同时保存原始四类概率。
OCT-C8 在读取时仅保留 CNV、DME、DRUSEN、NORMAL，不删除原文件。
其共同类别数量为 **10 / 7 / 86 / 1,443**，所以必须结合 macro-F1 和每类 recall，不能只看 accuracy。

仓库只存代码、配置与说明，不含图像、patient manifests、模型权重或私人 Drive 链接。
精确复现需要小组当前的清洗 ZIP；原始公开数据不能直接冒充这一清洗划分。
数据来源、目录结构与校验值见 [`docs/DATA.md`](docs/DATA.md)。

## 本地 / 服务器命令

使用 Python 3.10+，先按硬件安装 CUDA PyTorch，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
python src/prepare_data.py --oct2017 /path/OCT2017-cleaned.zip --mode smoke --out data/audit
python src/train.py --config configs/config.json --variant basic --seed 42 --out runs/smoke/basic_seed42 --smoke
```

完整数据准备使用一个新的目录：

```bash
python src/prepare_data.py --mode full --oct2017 /path/OCT2017-cleaned.zip --neh /path/NEH-cleaned.zip --oct-c8 /path/OCT-C8-cleaned.zip --out data/audit_full
```

将配置副本的 `audit_dir` 改为 `data/audit_full`。四组分别训练，最后一次性评价预先确定的全部方法：

```bash
python src/train.py --config configs/full.json --variant na_dynamic --seed 42 --out runs/full/na_dynamic_seed42 --resume --archive archive/full
# 同样运行 basic、dynamic、basic_noise；全部完成后：
python src/evaluate.py --root runs/full --config configs/full.json --experiments configs/experiments.json --archive archive/full
python src/aggregate.py --root runs/full
```

`configs/full.json` 是自行复制并修改的配置文件，不是仓库预置文件。
`--resume` 恢复相同配置、代码和数据下的未完成 run；已完成 run 不会覆盖。
测试前冻结 checkpoint，外部结果不参与训练或调参。
Slurm 示例见 [`scripts`](scripts)，路径通过环境变量提供，不依赖个人服务器账号。

## 文件说明

- `notebooks/OCT_Colab.ipynb`：Drive、数据准备、试跑、完整训练、恢复与评价入口。
- `src/models.py`、`src/noise.py`：模型和噪声损失。
- `src/prepare_data.py`：ZIP 检查、现有 split 读取、重复内容检查和缓存。
- `src/train.py`：训练、早停、checkpoint 和运行证据。
- `src/evaluate.py`、`src/aggregate.py`：冻结后的测试、指标、bootstrap 和汇总。
- `configs/`：训练参数、单次实验列表、数据校验值。
- `tests/`：合成数据与数学正确性测试，不包含真实医学图像。
- `docs/METHODS.md`：论文支持、实现假设和解释边界。
- `docs/ENVIRONMENT.md`：依赖、断线恢复和环境差异。
- `docs/VERIFICATION.md`：实际完成的检查及未验证的范围。

## References

1. Chutani D, Yalavarthy PK. *Noise-aware dynamic convolution for improved generalizability of retinal disease diagnosis using optical coherence tomography images*. JBO (2026). [DOI: 10.1117/1.JBO.31.8.086006](https://doi.org/10.1117/1.JBO.31.8.086006). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13464582/).
2. Chen Y, Dai X, Liu M, Chen D, Yuan L, Liu Z. *Dynamic Convolution: Attention over Convolution Kernels*. CVPR (2020). [Original paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Dynamic_Convolution_Attention_Over_Convolution_Kernels_CVPR_2020_paper.html).

本项目用于课程与研究复现。未报告临床验证或保证跨数据集性能提升。
代码使用了 AI 辅助开发，应由团队检查和理解，并按课程要求披露；仓库不代表任何成员的独立个人贡献。
