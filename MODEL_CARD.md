# Model Card for DINOv2-S/B/L/g

These are Vision Transformer models trained following the method described in the papers:
"DINOv2: Learning Robust Visual Features without Supervision"
and
"Vision Transformers Need Registers".

We provide 8 models: 
- 1 ViT-g trained from scratch with 3 ViT-S/B/L models distilled from the ViT-g, without registers.
- 1 ViT-g trained from scratch with 3 ViT-S/B/L models distilled from the ViT-g, with registers.

## Model Details
The model takes an image as input and returns a class token and patch tokens, and optionally 4 register tokens.

The embedding dimension is: 
- 384 for ViT-S.
- 768 for ViT-B.
- 1024 for ViT-L.
- 1536 for ViT-g.

The models follow a Transformer architecture, with a patch size of 14. In the case of registers, we add 4 register tokens, learned during training, to the input sequence after the patch embedding.

For a 224x224 image, this results in 1 class token + 256 patch tokens, and optionally 4 register tokens.

The models can accept larger images provided the image shapes are multiples of the patch size (14). 
If this condition is not verified, the model will crop to the closest smaller multiple of the patch size.

### Model Description

- **Developed by:** Meta AI
- **Model type:** Vision Transformer
- **License:** Apache License 2.0

- **Repository:** https://github.com/facebookresearch/dinov2
- **Paper:** https://arxiv.org/abs/2304.07193
- **Demo:** https://dinov2.metademolab.com/

## Uses

The models are vision backbones providing multi-purpose features for downstream tasks.

### Direct Use

The models can be used without fine-tuning, with downstream classifiers as simple as linear layers, to obtain competitive results:
- on depth estimation, semantic segmentation, using linear layers.
- on image classification, using k-NN classifiers on the class token.
- on image classification, with logistic regression classifiers applied on the class token.
- on image classification, with a linear layer applied on the class token and the average of the patch tokens.
- on image retrieval using nearest neighbors.

### Downstream Use

It is technically possible to perform fine-tuning on the models, for small gains (we measured +2% on ImageNet-1k classification). 
We recommend keeping this as a very last step and only when necessary, as the features already provide good performance out-of-the-box.

## Bias, Risks, and Limitations

Despite improvements thanks to the training method not using annotations, we still observe significant biases in our models toward rich households from Western countries.

### Recommendations

We expect fine-tuning will increase the biases in the features produced by the model as they will be tuned to the fine-tuning labels.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import torch

# DINOv2
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# DINOv2 with registers
dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
```

## Training Details

### Training Data

- **Training data:** LVD-142M (see paper)
- **Training regime:** fp16 using PyTorch-FSDP mixed-precision.

### Training Procedure 

- **Training objective:**
  - DINO self-distillation loss with multi-crop
  - iBOT masked-image modeling loss
  - KoLeo regularization on [CLS] tokens
- **Architectures:**
  - ViT-S (21M params): Patch size 14, embedding dimension 384, 6 heads, MLP FFN
  - ViT-B (86M params): Patch size 14, embedding dimension 768, 12 heads, MLP FFN
  - ViT-L (0.3B params): Patch size 14, embedding dimension 1024, 16 heads, MLP FFN
  - ViT-g (1.1B params): Patch size 14, embedding dimension 1536, 24 heads, SwiGLU FFN
- **Distillation:**
  - Distillation follows the standard DINOv2 pretraining procedure, except the teacher is a pretrained ViT-g, frozen.

## Evaluation

We refer users to the associated papers for the evaluation protocols.

<table>
  <tr>
    <th colspan="2"></th>
    <th colspan="3">ImageNet-1k</th>
    <th>NYU-Depth v2</th>
    <th>SUN-RGBD</th>
    <th>ADE20k</th>
    <th>iNaturalist 2018</th>
    <th>Oxford-H</th>
  </tr>
  <tr>
    <th rowspan="2">model</th>
    <th rowspan="2">with <br /> registers</th>
    <th>classif. (acc)</th>
    <th>classif. (acc)</th>
    <th>classif. V2 (acc)</th>
    <th>depth (RMSE)</th>
    <th>depth (RMSE)</th>
    <th>segm. (mAP)</th>
    <th>classif. (acc)</th>
    <th>retrieval (mAP)</th>
  </tr>
  <tr>
    <!-- <th>^</th> -->
    <th>k-NN</th>
    <th>linear</th>
    <th>linear</th>
    <th>linear<br />4 layers</th>
    <th>NYU-D transfer</th>
    <th>multiscale</th>
    <th>linear</th>
    <th>nearest neighbor</th>
  </tr>
  <tr>
    <td>ViT-S/14</td>
    <td align="center">:x:</td>
    <td align="right">79.0%</td>
    <td align="right">81.1%</td>
    <td align="right">70.8%</td> 
    <td align="right">0.417</td> 
    <td align="right">0.431</td> 
    <td align="right">47.2</td> 
    <td align="right">69.5%</td> 
    <td align="right">43.2</td> 
  </tr>
  <tr>
    <td>ViT-S/14</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">79.1%</td>
    <td align="right">80.9%</td>
    <td align="right">71.0%</td> 
    <td align="right">N/A</td> 
    <td align="right">N/A</td> 
    <td align="right">N/A</td> 
    <td align="right">67.6%</td> 
    <td align="right">39.5</td> 
  </tr>
  <tr>
    <td>ViT-B/14</td>
    <td align="center">:x:</td>
    <td align="right">82.1%</td>
    <td align="right">84.5%</td>
    <td align="right">74.9%</td>
    <td align="right">0.362</td> 
    <td align="right">0.400</td> 
    <td align="right">51.3</td> 
    <td align="right">76.3%</td> 
    <td align="right">49.5</td> 
  </tr>
    <td>ViT-B/14</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">82.0%</td>
    <td align="right">84.6%</td>
    <td align="right">75.6%</td>
    <td align="right">N/A</td> 
    <td align="right">N/A</td> 
    <td align="right">N/A</td> 
    <td align="right">73.8%</td> 
    <td align="right">51.0</td> 
  </tr>
  <tr>
    <td>ViT-L/14</td>
    <td align="center">:x:</td>
    <td align="right">83.5%</td>
    <td align="right">86.3%</td>
    <td align="right">77.6%</td>
    <td align="right">0.333</td> 
    <td align="right">0.396</td> 
    <td align="right">53.1</td> 
    <td align="right">79.8%</td> 
    <td align="right">54.0</td> 
  </tr>
  <tr>
    <td>ViT-L/14</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">83.8%</td>
    <td align="right">86.7%</td>
    <td align="right">78.5%</td>
    <td align="right">N/A</td> 
    <td align="right">N/A</td> 
    <td align="right">N/A</td> 
    <td align="right">80.9%</td> 
    <td align="right">55.7</td> 
  </tr>
  <tr>
    <td>ViT-g/14</td>
    <td align="center">:x:</td>
    <td align="right">83.5%</td>
    <td align="right">86.5%</td>
    <td align="right">78.4%</td>
    <td align="right">0.298</td> 
    <td align="right">0.362</td> 
    <td align="right">53.0</td> 
    <td align="right">81.6%</td> 
    <td align="right">52.3</td> 
  </tr>
  <tr>
  <tr>
    <td>ViT-g/14</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">83.7%</td>
    <td align="right">87.1%</td>
    <td align="right">78.8%</td>
    <td align="right">N/A</td> 
    <td align="right">N/A</td> 
    <td align="right">N/A</td> 
    <td align="right">81.5%</td> 
    <td align="right">58.2</td> 
  </tr>
</table>

## Environmental Impact

- **Hardware Type:** Nvidia A100
- **Hours used:** 22,000 for ViT-g, 4,500 for ViT-S distillation, 5,300 for ViT-B distillation, 8,000 for ViT-L distillation
- **Cloud Provider:** Private infra
- **Compute Region:** USA
- **Carbon Emitted:** 7t CO2eq

#### Hardware

Nvidia A100 GPUs

#### Software

PyTorch 2.0,
xFormers 0.0.18

**BibTeX**

```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```
