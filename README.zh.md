[English](README.md) | [简体中文](README.zh.md)

# ResNet18 口罩识别

基于 PyTorch 的图像二分类学习项目，识别**佩戴口罩**与**未佩戴口罩**两类图片。项目直接实现残差块和 ResNet18 风格网络，未加载 torchvision 预训练分类器。

仓库包含脚本、1,030 张数据集图片、已有的 `best_model.pth` 权重、示例图片和[训练曲线图](Figure_1.pdf)。这是图像分类练习，人脸定位、视频处理和线上应用不在实现范围内。本文不声明独立复现的准确率。

## 实现内容

- 带可选投影捷径的残差块，以及四个残差阶段。
- RGB 图片统一缩放至 224 × 224，使用固定参数归一化。
- 默认使用 Adam（`lr=0.001`）、交叉熵、32 的批次大小，训练 50 轮。
- 按验证集准确率选择最佳权重并绘制学习曲线。
- 测试集准确率评估和单张图片预测示例。

## 环境准备

原说明建议使用 Python 3.9 或更高版本。依赖没有固定版本，因此不代表经过验证的兼容范围。创建并激活虚拟环境后安装：

```bash
python -m venv .venv
# 在当前终端激活 .venv。
python -m pip install -r requirements.txt
```

依赖包括 PyTorch、torchvision、torchsummary、Pillow、NumPy、pandas 和 Matplotlib。请根据 CPU/CUDA 环境选择合适的 PyTorch 安装方式。脚本在 CUDA 可用时使用 GPU，否则使用 CPU。

命令应在仓库根目录执行。训练和测试使用 Windows 风格路径（`data\train`、`data\test`），其他操作系统需要调整。训练指定 Matplotlib 的 `TkAgg` 后端并弹出图表，需要桌面/Tk 环境；无界面运行时需调整绘图设置。

## 数据与运行流程

已跟踪的数据包含 `dataset/mask/` 中 500 张图片和 `dataset/no_mask/` 中 530 张图片。`ImageFolder` 按目录名确定类别索引，当前顺序为 `mask`、`no_mask`，应与预测标签保持一致。

### 1. 生成数据划分

```bash
python data_partitioning.py
```

脚本按类别将 90% 图片复制到 `data/train/`，10% 复制到 `data/test/`，未固定随机种子。它会保留目标目录的旧文件：在已有划分上重复运行，可能使同一图片同时出现在两组中。每次重新划分应使用全新的生成目录 `data/`。生成的副本已从 Git 中排除。

### 2. 训练或使用已有权重

```bash
python model_train.py
```

训练脚本使用未固定种子的 `random_split`，将 `data/train/` 再按 80%/20% 划分训练集和验证集。验证准确率最高的参数保存到 `best_model.pth`，会覆盖已有权重；训练结束后显示曲线。若需保留仓库中的权重，请事先另行备份。

### 3. 评估与预测

```bash
python model_test.py
```

脚本加载 `best_model.pth`，评估 `data/test/`，并预测 `images.jfif` 的类别。即使使用已有权重，也需先生成测试划分。预测其他 RGB 图片时可修改 `model_test.py` 中的图片路径。原说明记录已有权重约 43 MB，本文未独立核验其训练划分和效果。

查看网络结构摘要：

```bash
python model.py
```

## 目录说明

```text
dataset/                mask/no_mask 原始图片
data_partitioning.py    随机 9:1 复制划分
mean_std.py             归一化统计探索脚本
model.py                残差块与 ResNet18 风格分类器
model_train.py          训练、验证、保存权重和绘图
model_test.py           测试准确率与单图预测
best_model.pth          已有模型参数
Figure_1.pdf            已有训练曲线图
images.jfif             默认预测示例
```

## 复现限制与权利说明

`model_test.py` 是数据集评估脚本，并非自动化单元测试套件。本次文档更新未执行训练、加载权重或评估。随机划分会导致结果变化。

训练与测试使用相同的固定归一化参数。`mean_std.py` 属于探索工具：其逐通道统计除以数组全部元素数量，输出方差而非标准差。使用该脚本替换归一化参数前，应先检查计算方式。

仓库未注明原始数据集来源及许可，也未包含许可证文件。重新分发图片或权重前请核实相应权利；本仓库未提供明确的开源许可授权。
