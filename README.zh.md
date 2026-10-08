<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

:new: [2025-12-18] *支持加载 XRay-DINO 骨干网络，参考论文 [Advancing human-centric AI for robust X-ray analysis through holistic self-supervised learning](https://arxiv.org/pdf/2405.01469)，详情见[此处](#预训练骨干网络xray-dino)*

:new: [2025-12-16] *添加 Channel-Adaptive DINO 代码，参考论文 [Scaling Channel-Adaptive Self-Supervised Learning](https://openreview.net/forum?id=pT8sgtRVAf)，详情见[此处](#dinov2-生物学应用)*

:new: [2025-12-16] *添加 Cell-DINO 代码，参考论文 [Cell-DINO: Self-Supervised Image-based Embeddings for Cell Fluorescent Microscopy](to appear in Plos One Computational Biology)，详情见[此处](#dinov2-生物学应用)*

[2025-08-14] *请关注延续该技术路线的最新工作 [DINOv3](https://github.com/facebookresearch/dinov3)。*

[2025-06-11] *添加 dino.txt 推理代码，参考论文 [DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](https://arxiv.org/abs/2412.16334)。*

[2023-10-26] *添加带寄存器（registers）的 DINOv2 骨干网络，参考论文 [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)。*

# DINOv2: 无监督学习鲁棒视觉特征

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

Maxime Oquab,
Timothée Darcet,
Théo Moutakanni,
Huy V. Vo,
Marc Szafraniec,
Vasil Khalidov,
Patrick Labatut,
Armand Joulin,
Piotr Bojanowski

[[`论文 #1`](https://arxiv.org/abs/2304.07193)] [[`论文 #2`](https://arxiv.org/abs/2309.16588)] [[`博客`](https://ai.facebook.com/blog/dino-v2-computer-vision-self-supervised-learning/)] [[`在线演示`](https://dinov2.metademolab.com)] [[`BibTeX 引文`](#引用-dinov2)]

DINOv2 的 PyTorch 实现与预训练模型。更多详细信息请参阅论文：**[DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)** 与 **[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)**。

DINOv2 模型能够提取高性能的视觉特征，这些特征可以直接与线性层等简易分类器结合，应用于各种计算机视觉任务中；所提取的视觉特征具备极强的鲁棒性，无需任何微调即可跨领域表现优异。所有模型均在包含 1.42 亿张图像的数据集上进行预训练，全程不使用任何标签或标注。

https://github.com/facebookresearch/dinov2/assets/60359573/f168823e-7922-415a-b429-578badf5c356

<div align="center">
  所有视频帧 patch 特征前三个主成分的可视化展示（已映射至 RGB 颜色空间）。
</div>

## 预训练模型

<table style="margin: auto">
  <thead>
    <tr>
      <th>模型</th>
      <th>参数量</th>
      <th>带寄存器<br />(Registers)</th>
      <th>ImageNet<br />k-NN</th>
      <th>ImageNet<br />线性评估</th>
      <th>权重下载</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14 蒸馏版</td>
      <td align="right">21 M</td>
      <td align="center">:x:</td>
      <td align="right">79.0%</td>
      <td align="right">81.1%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth">仅骨干网络</a></td>
    </tr>
    <tr>
      <td>ViT-S/14 蒸馏版</td>
      <td align="right">21 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">79.1%</td>
      <td align="right">80.9%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth">仅骨干网络</a></td>
    </tr>
    <tr>
      <td>ViT-B/14 蒸馏版</td>
      <td align="right">86 M</td>
      <td align="center">:x:</td>
      <td align="right">82.1%</td>
      <td align="right">84.5%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth">仅骨干网络</a></td>
    </tr>
    <tr>
      <td>ViT-B/14 蒸馏版</td>
      <td align="right">86 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">82.0%</td>
      <td align="right">84.6%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth">仅骨干网络</a></td>
    </tr>
    <tr>
      <td>ViT-L/14 蒸馏版</td>
      <td align="right">300 M</td>
      <td align="center">:x:</td>
      <td align="right">83.5%</td>
      <td align="right">86.3%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth">仅骨干网络</a></td>
    </tr>
    <tr>
      <td>ViT-L/14 蒸馏版</td>
      <td align="right">300 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">83.8%</td>
      <td align="right">86.7%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth">仅骨干网络</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="center">:x:</td>
      <td align="right">83.5%</td>
      <td align="right">86.5%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth">仅骨干网络</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">83.7%</td>
      <td align="right">87.1%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth">仅骨干网络</a></td>
    </tr>
  </tbody>
</table>

### 预训练骨干网络（通过 PyTorch Hub）

请按照[此处](https://pytorch.org/get-started/locally/)的说明安装 PyTorch（加载模型的唯一必需依赖项）。强烈建议安装支持 CUDA 的 PyTorch 版本。

仓库中附带了对应的[模型卡片 (Model Card)](MODEL_CARD.md)。

```python
import torch

# DINOv2
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# 带寄存器的 DINOv2
dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
```

### 预训练骨干网络：XRay-DINO

模型下载申请地址：

https://ai.meta.com/resources/models-and-libraries/raydino-downloads/

填写表单后，您将收到一封包含临时下载链接的电子邮件。您可以使用 `wget` 下载模型并指定本地文件系统中的检查点路径，也可以直接在以下代码中使用邮件中的 URL：

```python
import torch

REPO_DIR = <PATH/TO/A/LOCAL/DIRECTORY/WHERE/THE/DINOV2/REPO/WAS/CLONED>

xray_dino_vitl16 = torch.hub.load(REPO_DIR, 'xray_dino_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
```

**许可证**  
模型权重遵循 FAIR 非商业研究许可证（FAIR Noncommercial Research License）发布。详见 LICENSE_XRAY_DINO_MODEL。

### 预训练任务头 - 图像分类

<table style="margin: auto">
  <thead>
    <tr>
      <th rowspan="2">骨干网络</th>
      <th rowspan="2">带寄存器<br />(Registers)</th>
      <th>权重下载</th>
    </tr>
    <tr>
      <th>ImageNet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14 蒸馏版</td>
      <td align="center">:x:</td>
      <td>
        线性头 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear4_head.pth">4 层</a>)
      </td>
    </tr>
    <tr>
      <td>ViT-S/14 蒸馏版</td>
      <td align="center">:white_check_mark:</td>
      <td>
        线性头 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_linear4_head.pth">4 层</a>)
      </td>
    </tr>
    <tr>
      <td>ViT-B/14 蒸馏版</td>
      <td align="center">:x:</td>
      <td>
        线性头 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear4_head.pth">4 层</a>)
    </tr>
    <tr>
      <td>ViT-B/14 蒸馏版</td>
      <td align="center">:white_check_mark:</td>
      <td>
        线性头 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_linear4_head.pth">4 层</a>)
      </td>
    </tr>
    <tr>
      <td>ViT-L/14 蒸馏版</td>
      <td align="center">:x:</td>
      <td>
        线性头 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear4_head.pth">4 层</a>)
      </td>
    </tr>
    <tr>
      <td>ViT-L/14 蒸馏版</td>
      <td align="center">:white_check_mark:</td>
      <td>
        线性头 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_linear4_head.pth">4 层</a>)
      </td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="center">:x:</td>
      <td>
        线性头 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear4_head.pth">4 层</a>)
      </td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="center">:white_check_mark:</td>
      <td>
        线性头 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_lreg4_inear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_linear4_head.pth">4 层</a>)
      </td>
    </tr>
  </tbody>
</table>

完整的分类器模型可以通过 PyTorch Hub 加载：

```python
import torch

# DINOv2
dinov2_vits14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
dinov2_vitb14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
dinov2_vitl14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
dinov2_vitg14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_lc')

# 带寄存器的 DINOv2
dinov2_vits14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
dinov2_vitb14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
dinov2_vitl14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc')
dinov2_vitg14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg_lc')
```

### 预训练任务头 - 深度估计

<table style="margin: auto">
  <thead>
    <tr>
      <th rowspan="2">骨干网络</th>
      <th colspan="2">任务头下载</th>
    </tr>
    <tr>
      <th>NYUd</th>
      <th>KITTI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14 蒸馏版</td>
      <td>
        线性 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_linear4_head.pth">4 层</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_dpt_head.pth">DPT</a>
      </td>
      <td>
        线性 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_kitti_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_kitti_linear4_head.pth">4 层</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_kitti_dpt_head.pth">DPT</a>
      </td>
    </tr>
    <tr>
      <td>ViT-B/14 蒸馏版</td>
      <td>
        线性 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/vitb14/dinov2_vitb14_nyu_linear4_head.pth">4 层</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_nyu_dpt_head.pth">DPT</a>
      </td>
      <td>
        线性 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_kitti_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_kitti_linear4_head.pth">4 层</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_kitti_dpt_head.pth">DPT</a>
      </td>
    </tr>
    <tr>
      <td>ViT-L/14 蒸馏版</td>
      <td>
        线性 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_nyu_linear4_head.pth">4 层</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_nyu_dpt_head.pth">DPT</a>
      </td>
      <td>
        线性 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_kitti_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_kitti_linear4_head.pth">4 层</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_kitti_dpt_head.pth">DPT</a>
      </td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td>
        线性 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_nyu_linear4_head.pth">4 层</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_nyu_dpt_head.pth">DPT</a>
      </td>
      <td>
        线性 (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_kitti_linear_head.pth">1 层</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_kitti_linear4_head.pth">4 层</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_kitti_dpt_head.pth">DPT</a>
      </td>
    </tr>
  </tbody>
</table>

### 预训练任务头 - 语义分割

<table style="margin: auto">
  <thead>
    <tr>
      <th rowspan="2">骨干网络</th>
      <th>模型下载</th>
      <th colspan="2">任务头下载</th>
    </tr>
    <tr>
      <th>ADE20K</th>
      <th>ADE20K</th>
      <th>VOC2012</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14 蒸馏版</td>
      <td></td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_ade20k_linear_head.pth">线性</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_ade20k_ms_head.pth">多尺度</a>
      </td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_voc2012_linear_head.pth">线性</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_voc2012_ms_head.pth">多尺度</a>
      </td>
    </tr>
    <tr>
      <td>ViT-B/14 蒸馏版</td>
      <td></td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_linear_head.pth">线性</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_ms_head.pth">多尺度</a>
      </td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_voc2012_linear_head.pth">线性</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_voc2012_ms_head.pth">多尺度</a>
      </td>
    </tr>
    <tr>
      <td>ViT-L/14 蒸馏版</td>
      <td></td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_ade20k_linear_head.pth">线性</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_ade20k_ms_head.pth">多尺度</a>
      </td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_voc2012_linear_head.pth">线性</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_voc2012_ms_head.pth">多尺度</a>
      </td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth">Mask2Former</a>
      </td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_ade20k_linear_head.pth">线性</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_ade20k_ms_head.pth">多尺度</a>
      </td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_voc2012_linear_head.pth">线性</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_voc2012_ms_head.pth">多尺度</a>
      </td>
    </tr>
  </tbody>
</table>

### 预训练任务头 - 基于 dino.txt 的零样本任务

<table style="margin: auto">
  <thead>
    <tr>
      <th rowspan="2">骨干网络</th>
      <th rowspan="2">带寄存器<br />(Registers)</th>
      <th>权重与资源下载</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-L/14 蒸馏版</td>
      <td align="center">:white_check_mark:</td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth">视觉任务头</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth">文本模型</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/thirdparty/bpe_simple_vocab_16e6.txt.gz">词表 (Vocabulary)</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/thirdparty/LICENSE">词表许可证</a>
      </td>
    </tr>
  </tbody>
</table>

完整的 dino.txt 模型可以通过 PyTorch Hub 加载：

```python
import torch

# DINOv2
dinov2_vitl14_reg4_dinotxt_tet1280d20h24l = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l')
```

## 安装指南

训练和评估代码需要 PyTorch 2.0 和 [xFormers](https://github.com/facebookresearch/xformers) 0.0.18，以及若干第三方依赖包。请注意，代码仅在指定版本下经过测试，并要求在 Linux 环境下运行。请按照以下说明安装训练和评估所需的全部依赖：

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **（推荐）** - 克隆本仓库，然后使用提供的环境配置文件创建并激活 `dinov2` conda 环境：

```shell
conda env create -f conda.yaml
conda activate dinov2
```

*[pip](https://pip.pypa.io/en/stable/getting-started/)* - 克隆本仓库，然后使用提供的 `requirements.txt` 安装依赖：

```shell
pip install -r requirements.txt
```

对于密集预测任务（深度估计与语义分割），还需要额外依赖（特定版本的 `mmcv` 和 `mmsegmentation`），已包含在 `extras` 依赖配置中：

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **（推荐）**：

```shell
conda env create -f conda-extras.yaml
conda activate dinov2-extras
```

*[pip](https://pip.pypa.io/en/stable/getting-started/)*：

```shell
pip install -r requirements.txt -r requirements-extras.txt
```

## 数据准备

### ImageNet-1k

数据集根目录下应包含以下文件与目录结构：

- `<ROOT>/test/ILSVRC2012_test_00000001.JPEG`
- `<ROOT>/test/[..]`
- `<ROOT>/test/ILSVRC2012_test_00100000.JPEG`
- `<ROOT>/train/n01440764/n01440764_10026.JPEG`
- `<ROOT>/train/[...]`
- `<ROOT>/train/n15075141/n15075141_9993.JPEG`
- `<ROOT>/val/n01440764/ILSVRC2012_val_00000293.JPEG`
- `<ROOT>/val/[...]`
- `<ROOT>/val/n15075141/ILSVRC2012_val_00049174.JPEG`
- `<ROOT>/labels.txt`

本仓库提供的数据集实现要求在 extra 目录下存在以下元数据文件：

- `<EXTRA>/class-ids-TRAIN.npy`
- `<EXTRA>/class-ids-VAL.npy`
- `<EXTRA>/class-names-TRAIN.npy`
- `<EXTRA>/class-names-VAL.npy`
- `<EXTRA>/entries-TEST.npy`
- `<EXTRA>/entries-TRAIN.npy`
- `<EXTRA>/entries-VAL.npy`

这些元数据文件可以通过运行以下 Python 代码（仅需执行一次）生成：

```python
from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```

注意：root 目录和 extra 目录不需要是不同的目录，可以指向同一位置。

### ImageNet-22k

请根据您的本地环境调整[数据集类](dinov2/data/datasets/image_net_22k.py)。

<br />

:warning: 运行接下来各节中的训练与评估命令时，需要确保 `dinov2` 包已包含在 Python 模块搜索路径中，即直接在命令前添加 `PYTHONPATH=.` 即可。

## 模型训练

### 快速配置：在 ImageNet-1k 上训练 DINOv2 ViT-L/16

在 SLURM 集群环境中使用 submitit 在 4 个 A100-80GB 节点（32 张 GPU）上运行 DINOv2 训练：

```shell
python dinov2/run/train/train.py \
    --nodes 4 \
    --config-file dinov2/configs/train/vitl16_short.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

训练时间约为 1 天，生成的检查点在 k-NN 评估中应达到 81.6%，在线性评估中达到 82.9%。

训练代码每隔 12,500 次迭代会将教师模型（teacher）的权重保存到 `eval` 文件夹中以便评估。

### 完整配置：在 ImageNet-22k 上训练 DINOv2 ViT-L/14

在 SLURM 集群环境中使用 submitit 在 12 个 A100-80GB 节点（96 张 GPU）上运行 DINOv2 训练：

```shell
python dinov2/run/train/train.py \
    --nodes 12 \
    --config-file dinov2/configs/train/vitl14.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet22k:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

训练时间约为 3.3 天，生成的检查点在 k-NN 评估中应达到 82.0%，在线性评估中达到 84.5%。

训练代码每隔 12,500 次迭代会将教师模型（teacher）的权重保存到 `eval` 文件夹中以便评估。

## 模型评估

训练代码会定期保存教师模型权重。为了评估模型性能，可在单节点上运行以下评估流程：

### ImageNet-1k 上的 k-NN 分类评估

```shell
python dinov2/run/eval/knn.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/knn \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### ImageNet-1k 上的逻辑回归分类评估

```shell
python dinov2/run/eval/log_regression.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/logreg \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### ImageNet-1k 上带数据增强的线性分类评估

```shell
python dinov2/run/eval/linear.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/linear \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

我们发布了评估不同模型所得到的权重：

<table style="margin: auto">
  <tr>
    <th>模型</th>
    <th>带寄存器<br />(Registers)</th>
    <th>ImageNet<br />top-1</th>
    <th>线性评估权重</th>
  </tr>
  <tr>
    <td>ViT-S/14 蒸馏版</td>
    <td align="center">:x:</td>
    <td align="right">81.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear_head.pth">线性头权重</a></td>
  </tr>
  <tr>
    <td>ViT-S/14 蒸馏版</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">80.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_linear_head.pth">线性头权重</a></td>
  </tr>
  <tr>
    <td>ViT-B/14 蒸馏版</td>
    <td align="center">:x:</td>
    <td align="right">84.5%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear_head.pth">线性头权重</a></td>
  </tr>
  <tr>
    <td>ViT-B/14 蒸馏版</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">84.4%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_linear_head.pth">线性头权重</a></td>
  </tr>
  <tr>
    <td>ViT-L/14 蒸馏版</td>
    <td align="center">:x:</td>
    <td align="right">86.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth">线性头权重</a></td>
  </tr>
  <tr>
    <td>ViT-L/14 蒸馏版</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">86.5%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_linear_head.pth">线性头权重</a></td>
  </tr>
  <tr>
    <td>ViT-g/14</td>
    <td align="center">:x:</td>
    <td align="right">86.5%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth">线性头权重</a></td>
  </tr>
  <tr>
    <td>ViT-g/14</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">87.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_linear_head.pth">线性头权重</a></td>
  </tr>
</table>

提供的预训练模型权重可以在 ImageNet-1k 上按如下方式评估：

```shell
python dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vitg14_pretrain.yaml \
    --pretrained-weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

## Notebook 交互教程

仓库中提供了若干 Jupyter Notebook，帮助社区轻松上手和使用模型与代码：

<ul>
  <li><a href="https://github.com/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb">深度估计 (Depth estimation)</a> - 如何通过 mmcv 将深度估计任务头与匹配的骨干网络结合加载与使用</li>
  <li><a href="https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb">语义分割 (Semantic segmentation)</a> - 如何通过 mmcv 将分割任务头与匹配的骨干网络结合加载与使用，以及如何加载和使用在 ADE20K 上训练的基于 Mask2Former 的分割模型</li>
</ul>

## 许可证

DINOv2 代码和模型权重采用 Apache License 2.0 许可证发布。详见 [LICENSE](LICENSE)。

## 贡献指南

请参阅[贡献指南](CONTRIBUTING.md)和[行为准则](CODE_OF_CONDUCT.md)。

## 引用 DINOv2

如果您觉得本项目对您的研究或工作有所帮助，请考虑给一个星标 :star: 并引用相关论文 :t-rex:：

```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```

```
@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```

```
@misc{jose2024dinov2meetstextunified,
  title={DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment},
  author={Cijo Jose and Théo Moutakanni and Dahyun Kang and Federico Baldassarre and Timothée Darcet and Hu Xu and Daniel Li and Marc Szafraniec and Michaël Ramamonjisoa and Maxime Oquab and Oriane Siméoni and Huy V. Vo and Patrick Labatut and Piotr Bojanowski},
  journal={arXiv:2412.16334},
  year={2024}
}
```

# DINOv2 生物学应用

`cell_dino` 文件夹中的源代码内容（包括代码和模型权重）仅供学术研究使用。严禁用于医疗程序，包括任何诊断、治疗或疗愈应用。请勿将该模型用于任何临床目的或替代专业医疗判断。

## 扩展通道自适应自监督学习 (Channel-Adaptive DINO)

 [[`论文`](https://openreview.net/forum?id=pT8sgtRVAf)] [[`BibTeX 引文`](#引用-channeladaptivedino-和-dinov2)]

Alice V. De Lorenci, Seungeun Yi, Théo Moutakanni, Piotr Bojanowski, Camille Couprie, Juan C. Caicedo, Wolfgang M. Pernice,

特别鸣谢 Elouan Gardes 对代码库做出的贡献。

 [README 文档](https://github.com/facebookresearch/dinov2/blob/main/docs/README_CHANNEL_ADAPTIVE_DINO.md)

## Cell-DINO：用于细胞荧光显微镜的自监督图像表征 (Cell-DINO)

Théo Moutakanni, Camille Couprie, Seungeun Yi, Elouan Gardes, Piotr Bojanowski, Hugo Touvron, Michael Doron, Zitong S. Chen, Nikita Moshkov, Mathilde Caron, Armand Joulin, Wolfgang M. Pernice, Juan C. Caicedo

即将正式发布。

 [README 文档](https://github.com/facebookresearch/dinov2/blob/main/docs/README_CELL_DINO.md)

## 预训练模型

ℹ️ 请点击下方提供的链接申请获取所有模型权重：通过后，您将收到一封包含所有可用模型权重完整下载链接列表的电子邮件。这些 URL 可用于：

- 将模型或适配器（adapter）权重下载到本地文件系统，并通过 `pretrained_path` 参数让 `torch.hub.load()` 指向这些本地权重；或者
- 直接调用 `torch.hub.load()` 并通过 `pretrained_url` 参数直接从 URL 下载并加载骨干网络。

⚠️ 请使用 `wget` 而不是网页浏览器来下载权重。

**下载链接：**  
https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/

```python
import torch

REPO_DIR = <PATH/TO/A/LOCAL/DIRECTORY/WHERE/THE/DINOV2/REPO/WAS/CLONED>

# 方式一：先下载 URL 链接，再加载本地权重：
cell_dino_vits8 = torch.hub.load(REPO_DIR, 'cell_dino_cp_vits8', source='local', pretrained_path=<CHECKPOINT/PATH>)
# 方式二：直接在 `torch.hub.load` 中传入 URL 自动下载加载：
cell_dino_vits8 = torch.hub.load(REPO_DIR, 'cell_dino_cp_vits8', source='local', pretrained_url=<CHECKPOINT/URL>)

# 其他模型类似：
cell_dino_vitl16_hpa_sc = torch.hub.load(REPO_DIR, 'cell_dino_hpa_vitl16', source='local', pretrained_path=<CHECKPOINT/PATH>)
cell_dino_vitl16_hpa_fov = torch.hub.load(REPO_DIR, 'cell_dino_hpa_vitl16', source='local', pretrained_path=<CHECKPOINT/PATH>)
channel_adaptive_dino_vitl16 = torch.hub.load(REPO_DIR, 'channel_adaptive_dino_vitl16', source='local', pretrained_path=<CHECKPOINT/PATH>)
cell_dino_vitl14 = torch.hub.load(REPO_DIR, 'cell_dino_hpa_vitl14', source='local', pretrained_path=<CHECKPOINT/PATH>)
```

## 许可证说明

代码基于 CC BY NC 许可证发布。详见 [LICENSE_CELL_DINO_CODE](LICENSE_CELL_DINO_CODE)。  
模型权重基于 FAIR 非商业研究许可证（FAIR Noncommercial Research License）发布。详见 [LICENSE_CELL_DINO_CODE_WEIGHTS](LICENSE_CELL_DINO_CODE_WEIGHTS)。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年10月08日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
