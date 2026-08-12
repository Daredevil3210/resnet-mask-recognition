# 基于 ResNet18 的口罩识别

这是一个使用 PyTorch 从零实现 ResNet18 的二分类项目，用于识别图片中的人物是否佩戴口罩。仓库包含训练与测试代码、原始数据集、示例图片，以及已训练好的模型权重。

## 项目结构

```text
.
├── dataset/              # 原始数据集：mask / no_mask
├── data_partitioning.py  # 按 9:1 生成训练集和测试集
├── mean_std.py           # 计算数据集归一化参数
├── model.py              # ResNet18 网络结构
├── model_train.py        # 模型训练与曲线绘制
├── model_test.py         # 测试集评估与单张图片预测
├── best_model.pth        # 已训练的模型权重
└── Figure_1.pdf          # 训练结果图
```

`data/` 是由原始数据集自动生成的副本，因此不会提交到仓库。运行数据划分脚本即可重新创建。

## 环境安装

建议使用 Python 3.9 或更高版本，并在虚拟环境中安装依赖：

```bash
pip install -r requirements.txt
```

如果安装 PyTorch 时需要指定 CUDA 版本，请参考 PyTorch 官网给出的安装命令。

## 使用方法

1. 生成训练集和测试集：

   ```bash
   python data_partitioning.py
   ```

2. 训练模型：

   ```bash
   python model_train.py
   ```

   脚本会自动使用可用的 CUDA 设备，否则使用 CPU。训练完成后，最优权重保存为 `best_model.pth`。

3. 测试模型：

   ```bash
   python model_test.py
   ```

   默认会评估 `data/test`，并对项目中的 `images.jfif` 进行单张图片预测。可在 `model_test.py` 中替换为其他图片路径。

4. 查看网络结构：

   ```bash
   python model.py
   ```

## 数据集

数据集包含两个类别：

- `mask`：佩戴口罩
- `no_mask`：未佩戴口罩

当前共 1,030 张图片，其中 `mask` 500 张、`no_mask` 530 张。

## 说明

- 输入图片统一缩放为 224 × 224。
- 使用 Adam 优化器与交叉熵损失函数。
- 默认训练 50 个 epoch，并保存验证集准确率最高的模型。
- 已训练权重约 43 MB，可直接用于测试。

