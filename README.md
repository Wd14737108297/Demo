# 改进 Faster R-CNN 在太阳能电池板缺陷检测中的应用与性能优化

PyTorch 实现 | 论文处于投刊阶段，标题：《改进 Faster R-CNN 在太阳能电池板缺陷检测中的应用与性能优化》

提出融合 **CBAM 卷积块注意力机制** 与 **Focal Loss** 的改进 Faster R-CNN，实现太阳能电池板常见缺陷与健康面板的高精度检测，辅助光伏组件自动化质检。

- GitHub 仓库：https://github.com/Wd14737108297/Improved-Faster-R-CNN-for-Solar-Panel-Defect-Detection

---

## 📦 资源下载

| 资源 | 下载方式 | 放置位置 |
|---|---|---|
| 数据集 | 百度网盘：[链接](https://pan.baidu.com/s/1Ok_INdRfG9HW6rtlcxHlAA?pwd=nqf3) 提取码：nqf3 | 解压后得到 `VOCdevkit/`，放项目根目录 |
| 预训练权重 + 训练权重 | 百度网盘：[链接](https://pan.baidu.com/s/1Hwcfb7of3KqYPwfUqsZSuA?pwd=y598) 提取码：y598 | 按压缩包内 `使用说明.txt` 放置 |
| 公开数据集 ELPV | https://gitcode.com/gh_mirrors/el/elpv-dataset | 按需下载 |

> 若网盘链接失效，请在本仓库提交 Issue 或联系邮箱 wudile@huuc.edu.cn 获取最新链接。

---

## 1. 研究背景与模型定位

太阳能电池板作为光伏系统的核心组件，其制造与运行过程中可能产生隐裂、断栅、热斑、污损等缺陷。传统人工巡检受主观经验影响显著，难以满足户外大规模光伏电站的实时监测需求。现有深度学习方法在复杂光照、沙尘覆盖、积雪遮挡等场景下适应性不足，微小缺陷识别精度仍有提升空间。

本文提出一种融合 CBAM 与 Focal Loss 的改进 Faster R-CNN 高精度缺陷检测方法。在 ResNet-50 骨干网络后嵌入 CBAM 模块，通过通道注意力与空间注意力的串行协同，实现对缺陷关键区域的自适应特征增强与复杂背景噪声的有效抑制；在 RPN 阶段引入 Focal Loss 替代传统交叉熵分类损失，通过动态调制难易样本损失权重，提升模型对隐裂、断栅、弱热斑等小尺度、弱纹理缺陷的挖掘能力与召回率。

在自建光伏板缺陷数据集上，改进模型的 **mAP50 达到 96.4%**，整体检测准确率为 **96.84%**，在精确率、召回率及 F1 分数等核心指标上均显著优于基准 Faster R-CNN 与主流对比模型，同时保持满足工业现场部署要求的推理速度。

---

## 2. 核心创新点

### 2.1 CBAM 注意力机制增强特征表达

在 Faster R-CNN 的骨干网络输出后嵌入卷积块注意力模块（CBAM），通过通道注意力与空间注意力两个子模块的串行计算，自适应增强缺陷关键区域的特征响应。

- 通道注意力对 1024 个特征通道的重要性进行重标定；
- 空间注意力对 38×38 空间位置进行权重再分配；
- 双重机制协同抑制户外复杂背景及光照变化带来的噪声干扰，显著提升缺陷识别的鲁棒性。

> 注意：论文中 CBAM 的空间注意力使用 7×7 卷积；本仓库 `nets/attention.py` 默认实现为 3×3 卷积。如需严格复现论文，请将空间注意力卷积改为 `kernel_size=7, padding=3`。

### 2.2 基于 Focal Loss 的 RPN 分类损失优化

针对太阳能电池板缺陷检测中正负样本严重不平衡问题，本文引入 Focal Loss 替代 RPN 分类分支的传统交叉熵损失。Focal Loss 通过聚焦参数 γ 和平衡因子 α，动态降低易分类样本的损失权重，迫使模型集中学习难例缺陷样本，从而有效缓解类别不均衡导致的检测偏向性问题。

本仓库中 Focal Loss 位于 `nets/frcnn_training.py`，默认参数：

```python
FocalLoss(alpha=0.5, gamma=2)
```

### 2.3 分阶段训练策略与多数据增强

采用冻结骨干与全参数微调相结合的双阶段训练策略，配合余弦退火学习率调度与混合精度训练，在保证收敛稳定性的同时提升训练效率。训练阶段引入随机缩放、水平翻转、HSV 色域扰动等数据增强手段，增强模型对不同光照条件与拍摄角度的泛化能力。

---

## 3. 实验数据集

### 3.1 自建光伏板缺陷数据集

实验数据来源于中国部分光伏电站的实地采集，包含健康面板及缺陷面板。数据集共包含 **6079 个样本**，并采用 **3:1:1** 的比例划分为训练集、验证集与测试集。

| 类别 | 训练集 | 验证集 | 测试集 | 总数 |
|---|---:|---:|---:|---:|
| Broken | 1048 | 353 | 303 | 1704 |
| Clean | 1255 | 420 | 455 | 2130 |
| Dirty | 1344 | 434 | 458 | 2245 |
| **Total** | **3647** | **1216** | **1216** | **6079** |

实际类别名以 `model_data/cls_classes.txt` 为准，当前为：

```text
Broken
Clean
Dirty
```

> 说明：论文中同时提到隐裂、断栅、热斑、污损等缺陷类型；本仓库实际检测类别为 Broken、Clean、Dirty 三类。若需扩展其他缺陷类别，请补充数据集并修改 `model_data/cls_classes.txt`。

**数据集下载：**

由于数据集体积较大，未直接上传至 GitHub，请通过百度网盘下载：

- **压缩包名称**：`VOC07+12+test.zip`
- **百度网盘链接**：https://pan.baidu.com/s/1Ok_INdRfG9HW6rtlcxHlAA?pwd=nqf3
- **提取码**：nqf3

**放置方式：**

下载后解压，将得到的 `VOCdevkit` 文件夹放在项目根目录下，最终目录结构如下：

```text
项目根目录/
├── VOCdevkit/
│   └── VOC2007/
│       ├── Annotations/
│       ├── JPEGImages/
│       └── ImageSets/
│           └── Main/
│               ├── train.txt
│               ├── val.txt
│               ├── trainval.txt
│               └── test.txt
├── model_data/
├── nets/
├── utils/
├── train.py
└── ...
```

> 如果下载的数据集已经包含划分好的 `train.txt`、`val.txt`、`trainval.txt`、`test.txt`，可直接使用；否则请运行 `python voc_annotation.py` 重新生成。

### 3.2 公开数据集 ELPV

为进一步提升模型泛化能力，本研究额外引入 ELPV Dataset 进行训练与评估，包含 2624 张单晶与多晶标注图像：

https://gitcode.com/gh_mirrors/el/elpv-dataset

---

## 4. 实验环境配置

### 4.1 论文实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 64-bit |
| GPU | RTX 4060 Laptop |
| CPU | Intel Core i7-14650HX |
| 内存 | 16GB |
| 深度学习框架 | PyTorch 2.8.0 |
| 编辑器 | PyCharm 2024.3.6 |
| 编程语言 | Python 3.9.23 |

### 4.2 依赖安装

推荐使用 Anaconda 创建虚拟环境：

```bash
conda create -n frcnn-pv-defect python=3.9
conda activate frcnn-pv-defect
```

安装 PyTorch 与 torchvision：

```bash
pip install torch torchvision
```

安装其他依赖：

```bash
pip install numpy opencv-python pillow matplotlib scipy tqdm tensorboard
pip install pycocotools thop torchsummary
```

> 注意：`get_map.py` 的 `map_mode=4` 使用 COCO 评估流程，需要安装 `pycocotools`。

### 4.3 硬件要求

- GPU：推荐 NVIDIA GPU，显存 ≥ 8GB；
- CPU：支持推理测试，但训练耗时显著增加，不推荐用于完整训练流程。

---

## 5. 实验结果

### 5.1 核心指标

| 指标 | 结果 |
|---|---:|
| mAP50 | 96.4% |
| Accuracy | 96.84% |
| 对比基准 Faster R-CNN | 显著提升 |
| 推理速度 | 与原始模型基本持平 |

### 5.2 与主流方法对比

| 方法 | 数据集 | mAP |
|---|---|---:|
| ESD-YOLOv8 | Pierdicca | 91.80% |
| Improved YOLOv5s | elpv | 94.30% |
| RMN-YOLO | PaddlePaddle | 90.50% |
| WCC-YOLO | PVEL-AD | 94.00% |
| Improved Mask R-CNN | ELPV | 89.00% |
| **Proposed Improved Faster R-CNN** | **Our dataset** | **96.40%** |

### 5.3 消融实验

完整改进模型（Faster R-CNN + CBAM + Focal Loss）在全部五项指标上均取得最高值：

- 相比原始 Faster R-CNN，Sensitivity 提升约 **5.8 个百分点**；
- Precision 提升约 **4.2 个百分点**；
- F1-score 与 Accuracy 同步提升；
- CBAM 与 Focal Loss 形成良好互补，共同提升多尺度、多类型缺陷检测能力。

---

## 6. 代码使用说明

### 6.1 克隆仓库

```bash
git clone https://github.com/Wd14737108297/Improved-Faster-R-CNN-for-Solar-Panel-Defect-Detection.git
cd Improved-Faster-R-CNN-for-Solar-Panel-Defect-Detection
```

### 6.2 生成训练标注文件

修改 `voc_annotation.py`：

```python
annotation_mode = 0
classes_path = 'model_data/cls_classes.txt'
trainval_percent = 0.8
train_percent = 0.75
VOCdevkit_path = 'VOCdevkit'
```

运行：

```bash
python voc_annotation.py
```

运行后会生成：

```text
2007_train.txt
2007_val.txt
```

### 6.3 模型训练

修改 `train.py` 中的关键参数：

```python
classes_path = 'model_data/cls_classes.txt'
model_path = 'model_data/voc_weights_resnet.pth'
input_shape = [600, 600]
backbone = "resnet50"
anchors_size = [8, 16, 32]

train_annotation_path = 'path/to/2007_train.txt'
val_annotation_path = 'path/to/2007_val.txt'
```

> 论文表 3 中最佳超参数为学习率 `0.01`、优化器 `SGD`。`train.py` 默认使用 Adam 且 `Init_lr=1e-5`，复现论文时请按需修改：
>
> ```python
> optimizer_type = "sgd"
> Init_lr = 1e-2
> Min_lr = 1e-4
> ```

运行训练：

```bash
python train.py
```

训练日志与权重默认保存在 `logs/` 目录。

### 6.4 模型评估

修改 `get_map.py`：

```python
map_mode = 4
classes_path = 'model_data/cls_classes.txt'
VOCdevkit_path = 'VOCdevkit'
map_out_path = 'map_out'
```

运行：

```bash
python get_map.py
```

输出包括：

- mAP50
- mAP50-95
- 全局 Precision、Recall、F1-score

> 注意：`map_mode=4` 使用 COCO 评估流程，需要 `map_out/ground-truth` 与 `map_out/detection-results` 中已有预测结果和真实标签。

### 6.5 模型预测

修改 `frcnn.py`：

```python
"model_path"   : 'logs/best_epoch_weights.pth',
"classes_path" : 'model_data/cls_classes.txt',
"backbone"     : "resnet50",
```

运行：

```bash
python predict.py
```

`predict.py` 支持以下模式：

```python
mode = "predict"      # 单张图片预测
mode = "video"        # 视频 / 摄像头检测
mode = "fps"          # 测试 FPS
mode = "dir_predict"  # 文件夹批量预测
```

### 6.6 查看网络结构

运行：

```bash
python summary.py
```

可查看模型结构与参数量、计算量。注意根据需要修改 `num_classes` 与 `backbone`。

### 6.7 预训练权重

由于 GitHub 文件大小限制，预训练权重与训练好的模型权重已上传至百度网盘，请下载后按说明放置。

**权重压缩包下载：**

- **压缩包名称**：`fasterrcnn_solar_panel_weights.zip`
- **百度网盘链接**：https://pan.baidu.com/s/1Hwcfb7of3KqYPwfUqsZSuA?pwd=y598
- **提取码**：y598

**压缩包内文件及放置位置：**

| 文件 | 放置目录 | 用途 |
|---|---|---|
| `voc_weights_resnet.pth` | `model_data/` | 训练预训练权重，用于加载整个 Faster R-CNN 模型参数 |
| `resnet50-19c8e357.pth` | `model_data/` | ResNet-50 主干 ImageNet 预训练权重（备用） |
| `best_epoch_weights.pth` | `logs/` | 训练好的最优模型权重，用于预测和 mAP 评估 |

> 压缩包内附有 `使用说明.txt`，可对照放置。若不想下载 `resnet50-19c8e357.pth`，代码也会自动从 PyTorch 官网下载，但国内网络可能较慢，建议直接使用压缩包内的文件。

**快速验证：**

放置完成后，运行单张图片预测：

```bash
python predict.py
```

评估 mAP：

```bash
python get_map.py
```

---

## 7. 项目文件结构

```text
.
├── nets/
│   ├── attention.py          # CBAM 注意力机制
│   ├── frcnn.py              # Faster R-CNN 模型定义
│   ├── frcnn_training.py     # Focal Loss 与训练器
│   ├── rpn.py                # 区域建议网络
│   ├── classifier.py         # RoI Head
│   ├── resnet50.py           # ResNet-50 骨干
│   └── vgg16.py              # VGG16 骨干
├── utils/
│   ├── anchors.py            # 先验框生成
│   ├── callbacks_enhanced.py # 训练评估回调
│   ├── dataloader.py         # 数据集与数据增强
│   ├── utils.py              # 通用工具
│   ├── utils_bbox.py         # 边界框解码
│   ├── utils_fit.py          # 训练流程
│   └── utils_map.py          # mAP 评估
├── model_data/
│   ├── cls_classes.txt
│   ├── simhei.ttf
│   └── voc_weights_resnet.pth
├── logs/
├── frcnn.py                  # 预测封装类
├── get_map.py                # mAP 评估脚本
├── predict.py                # 预测脚本
├── summary.py                # 模型结构查看
├── train.py                  # 训练脚本
└── voc_annotation.py         # VOC 数据标注生成
```

---

## 8. 已知问题与注意事项

1. **学习率与优化器**  
   论文表 3 中最佳超参数为学习率 `0.01`、优化器 `SGD`。`train.py` 默认使用 Adam 且 `Init_lr=1e-5`，复现论文时请按需修改。

2. **CBAM 空间卷积核**  
   论文描述空间注意力使用 7×7 卷积，本仓库 `nets/attention.py` 默认是 3×3。如需严格对齐论文，请修改为：

   ```python
   nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
   ```

3. **Focal Loss 参数**  
   `nets/frcnn_training.py` 中默认：

   ```python
   FocalLoss(alpha=0.5, gamma=2)
   ```

   可根据数据集调整 `alpha` 与 `gamma`。

4. **Cutout 数据增强**  
   `utils/dataloader.py` 中的 Cutout 增强代码位于 `return image_data, box` 之后，当前不会执行。若需启用，请将相关代码移动到 `return` 之前。

5. **路径不要包含空格**  
   VOC 数据集路径与图片名称中不要包含空格，否则可能影响训练与评估。

6. **显存不足**  
   若出现 OOM，请减小 `Freeze_batch_size` 与 `Unfreeze_batch_size`。

7. **训练中评估曲线仅供参考**  
   训练过程中的 mAP 曲线用于观察趋势，最终论文指标建议以 `get_map.py` 的输出为准。

8. **数据集适配**  
   当前模型与权重仅针对 `model_data/cls_classes.txt` 中的类别。若新增缺陷类别，需补充对应数据集并重新训练。数据集请从百度网盘下载，解压后放在项目根目录。

9. **图像分辨率**  
   输入图像会自动 resize 至 600×600 左右，若原始图像分辨率过低，可能导致早期微小缺陷特征丢失，建议输入图像分辨率 ≥ 600×600。

10. **CUDA 版本问题**  
    若安装 PyTorch 时出现 CUDA 不兼容，可替换为 CPU 版本，但训练效率会大幅下降。

---

## 9. 引用与联系方式

### 9.1 引用方式

论文处于投刊阶段，正式发表后将更新 BibTeX 引用格式，当前可临时引用：

```bibtex
@article{wu2026fasterrcnn,
  title={改进 Faster R-CNN 在太阳能电池板缺陷检测中的应用与性能优化},
  author={[作者姓名，待发表时补充]},
  journal={[期刊名称，待录用后补充]},
  year={2026},
  note={Manuscript submitted for publication}
}
```

### 9.2 联系方式

- 邮箱：wudile@huuc.edu.cn
- GitHub Issue：直接在本仓库提交 Issue，会在 1-3 个工作日内回复。

---
