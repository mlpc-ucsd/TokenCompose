<p align="center">

  <h1 align="center"><a href="https://mlpc-ucsd.github.io/TokenCompose/">🧩 TokenCompose</a>: Grounding Diffusion with Token-level Supervision</h1>
  <p align="center">
    <a href="https://zwcolin.github.io/"><strong>Zirui Wang</strong></a><sup>1, 3</sup>
    ·
    <a href="https://github.com/JamesSand"><strong>Zhizhou Sha</strong></a><sup>2, 3</sup>
    ·
    <a href="https://github.com/zh-ding"><strong>Zheng Ding</strong></a><sup>3</sup>
    ·
    <a href="https://github.com/modric197"><strong>Yilin Wang</strong></a><sup>2, 3</sup>
    ·
    <a href="https://pages.ucsd.edu/~ztu/"><strong>Zhuowen Tu</strong></a><sup>3</sup>
  </p>
  <p align="center">
    <sup>1</sup><strong>Princeton University</strong>
    ·
    <sup>2</sup><strong>Tsinghua University</strong>
    ·
    <sup>3</sup><strong>University of California, San Diego</strong>
  </p>

</p>

<h3 align="center">
  <a href="https://mlpc-ucsd.github.io/TokenCompose/"><strong>Project Page</strong></a>
  |
  <a href="https://mlpc-ucsd.github.io/TokenCompose/"><strong>arXiv</strong></a>
  |
  <a href="https://mlpc-ucsd.github.io/TokenCompose/"><strong>Twitter</strong></a>
</h3>

https://github.com/mlpc-ucsd/TokenCompose/assets/59942464/93feea16-4eac-49c3-b286-ee390a325b17

<p align="center">
  A <span style="color: lightblue">Stable Diffusion</span> model finetuned with <strong>token-level grounding objectives</strong> for enhanced <strong>multi-category instance composition</strong> and <strong>photorealism</strong>.
</p>

<br>

<div align="center">
  <img src="asset/teaser.jpg" alt="Logo" width="100%">
</div>



<table>

  <tr>
    <th rowspan="3" align="center">Method</th>
    <th colspan="9" align="center">Multi-category Instance Composition</th>
    <th colspan="2" align="center">Photorealism</th>
    <th colspan="1" align="center">Efficiency</th>
  </tr>

  <tr>
    <!-- <th align="center">&nbsp;</th> -->
    <th rowspan="2" align="center">Object Accuracy</th>
    <th colspan="4" align="center">COCO</th>
    <th colspan="4" align="center">ADE20K</th>
    <th rowspan="2" align="center">FID (COCO)</th>
    <th rowspan="2" align="center">FID (Flickr30K)</th>
    <th rowspan="2" align="center">Latency</th>
  </tr>

  <tr>
    <!-- <th align="center">&nbsp;</th> -->
    <th align="center">MG2</th>
    <th align="center">MG3</th>
    <th align="center">MG4</th>
    <th align="center">MG5</th>
    <th align="center">MG2</th>
    <th align="center">MG3</th>
    <th align="center">MG4</th>
    <th align="center">MG5</th>
  </tr>

  <tr>
    <td align="center"><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">SD 1.4</a></td>
    <td align="center">29.86</td>
    <td align="center">90.72<sub>1.33</sub></td>
    <td align="center">50.74<sub>0.89</sub></td>
    <td align="center">11.68<sub>0.45</sub></td>
    <td align="center">0.88<sub>0.21</sub></td>
    <td align="center">89.81<sub>0.40</sub></td>
    <td align="center">53.96<sub>1.14</sub></td>
    <td align="center">16.52<sub>1.13</sub></td>
    <td align="center">1.89<sub>0.34</sub></td>
    <td align="center"><u>20.88</u></td>
    <td align="center"><u>71.46</u></td>
    <td align="center"><b>7.54</b><sub>0.17</sub></td>
  </tr>

  <tr>
    <td align="center"><a href="https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch">Composable</a></td>
    <td align="center">27.83</td>
    <td align="center">63.33<sub>0.59</sub></td>
    <td align="center">21.87<sub>1.01</sub></td>
    <td align="center">3.25<sub>0.45</sub></td>
    <td align="center">0.23<sub>0.18</sub></td>
    <td align="center">69.61<sub>0.99</sub></td>
    <td align="center">29.96<sub>0.84</sub></td>
    <td align="center">6.89<sub>0.38</sub></td>
    <td align="center">0.73<sub>0.22</sub></td>
    <td align="center">-</td>
    <td align="center">75.57</td>
    <td align="center">13.81<sub>0.15</sub></td>
  </tr>

  <tr>
    <td align="center"><a href="https://github.com/silent-chen/layout-guidance">Layout</a></td>
    <td align="center">43.59</td>
    <td align="center">93.22<sub>0.69</sub></td>
    <td align="center">60.15<sub>1.58</sub></td>
    <td align="center">19.49<sub>0.88</sub></td>
    <td align="center">2.27<sub>0.44</sub></td>
    <td align="center"><u>96.05</u><sub>0.34</sub></td>
    <td align="center"><u>67.83</u><sub>0.90</sub></td>
    <td align="center">21.93<sub>1.34</sub></td>
    <td align="center">2.35<sub>0.41</sub></td>
    <td align="center">-</td>
    <td align="center">74.00</td>
    <td align="center">18.89<sub>0.20</sub></td>
  </tr>

  <tr>
    <td align="center"><a href="https://github.com/weixi-feng/Structured-Diffusion-Guidance">Structured</a></td>
    <td align="center">29.64</td>
    <td align="center">90.40<sub>1.06</sub></td>
    <td align="center">48.64<sub>1.32</sub></td>
    <td align="center">10.71<sub>0.92</sub></td>
    <td align="center">0.68<sub>0.25</sub></td>
    <td align="center">89.25<sub>0.72</sub></td>
    <td align="center">53.05<sub>1.20</sub></td>
    <td align="center">15.76<sub>0.86</sub></td>
    <td align="center">1.74<sub>0.49</sub></td>
    <td align="center">21.13</td>
    <td align="center">71.68</td>
    <td align="center"><u>7.74</u><sub>0.17</sub></td>
  </tr>

  <tr>
    <td align="center"><a href="https://github.com/yuval-alaluf/Attend-and-Excite">Attn-Exct</a></td>
    <td align="center"><u>45.13</u></td>
    <td align="center"><u>93.64</u><sub>0.76</sub></td>
    <td align="center"><u>65.10</u><sub>1.24</sub></td>
    <td align="center"><u>28.01</u><sub>0.90</sub></td>
    <td align="center"><b>6.01</b><sub>0.61</sub></td>
    <td align="center">91.74<sub>0.49</sub></td>
    <td align="center">62.51<sub>0.94</sub></td>
    <td align="center"><u>26.12</u><sub>0.78</sub></td>
    <td align="center"><u>5.89</u><sub>0.40</sub></td>
    <td align="center">-</td>
    <td align="center">71.68</td>
    <td align="center">25.43<sub>4.89</sub></td>
  </tr>

  <tr>
    <td align="center"><a href="https://github.com/mlpc-ucsd/TokenCompose"><strong>TokenCompose (Ours)</strong></a></td>
    <td align="center"><b>52.15</b></td>
    <td align="center"><b>98.08</b><sub>0.40</sub></td>
    <td align="center"><b>76.16</b><sub>1.04</sub></td>
    <td align="center"><b>28.81</b><sub>0.95</sub></td>
    <td align="center"><u>3.28</u><sub>0.48</sub></td>
    <td align="center"><b>97.75</b><sub>0.34</sub></td>
    <td align="center"><b>76.93</b><sub>1.09</sub></td>
    <td align="center"><b>33.92</b><sub>1.47</sub></td>
    <td align="center"><b>6.21</b><sub>0.62</sub></td>
    <td align="center"><b>20.19</b></td>
    <td align="center"><b>71.13</b></td>
    <td align="center"><b>7.56</b><sub>0.14</sub></td>
  </tr>

</table>



## 🆕 Models

| Stable Diffusion Version | Checkpoint 1 | Checkpoint 2 |
|:------------------------:|:------------:|:------------:|
| v1.4                     | [TokenCompose_SD14_A](https://huggingface.co/mlpc-lab/TokenCompose_SD14_A)         | [TokenCompose_SD14_B](https://huggingface.co/mlpc-lab/TokenCompose_SD14_B)         |
| v2.1                     | [TokenCompose_SD21_A](https://huggingface.co/mlpc-lab/TokenCompose_SD21_A)         | [TokenCompose_SD21_B](https://huggingface.co/mlpc-lab/TokenCompose_SD21_B)         |

Our finetuned models do not contain any extra modules and can be directly used in a standard diffusion model library (e.g., HuggingFace's Diffusers) by replacing the pretrained U-Net with our finetuned U-Net in a plug-and-play manner. We provide a [demo jupyter notebook](notebooks/example_usage.ipynb) which uses our model checkpoint to generate images. 

You can also use the following code to download our checkpoints and generate images:

```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "mlpc-lab/TokenCompose_SD14_A"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

prompt = "A cat and a wine glass"
image = pipe(prompt).images[0]  
    
image.save("cat_and_wine_glass.png")
```

## 📊 MultiGen 

See [MultiGen](multigen/readme.md) for details.




<table>
  <tr>
    <th rowspan="2" align="center">Method</th>
    <th colspan="4" align="center">COCO</th>
    <th colspan="4" align="center">ADE20K</th>
  </tr>

  <tr>
    <!-- <th align="center">&nbsp;</th> -->
    <th align="center">MG2</th>
    <th align="center">MG3</th>
    <th align="center">MG4</th>
    <th align="center">MG5</th>
    <th align="center">MG2</th>
    <th align="center">MG3</th>
    <th align="center">MG4</th>
    <th align="center">MG5</th>
  </tr>

  <tr>
    <td align="center"><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">SD 1.4</a></td>
    <td align="center">90.72<sub>1.33</sub></td>
    <td align="center">50.74<sub>0.89</sub></td>
    <td align="center">11.68<sub>0.45</sub></td>
    <td align="center">0.88<sub>0.21</sub></td>
    <td align="center">89.81<sub>0.40</sub></td>
    <td align="center">53.96<sub>1.14</sub></td>
    <td align="center">16.52<sub>1.13</sub></td>
    <td align="center">1.89<sub>0.34</sub></td>
  </tr>

  <tr>
    <td align="center"><a href="https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch">Composable</a></td>
    <td align="center">63.33<sub>0.59</sub></td>
    <td align="center">21.87<sub>1.01</sub></td>
    <td align="center">3.25<sub>0.45</sub></td>
    <td align="center">0.23<sub>0.18</sub></td>
    <td align="center">69.61<sub>0.99</sub></td>
    <td align="center">29.96<sub>0.84</sub></td>
    <td align="center">6.89<sub>0.38</sub></td>
    <td align="center">0.73<sub>0.22</sub></td>
  </tr>

  <tr>
    <td align="center"><a href="https://github.com/silent-chen/layout-guidance">Layout</a></td>
    <td align="center">93.22<sub>0.69</sub></td>
    <td align="center">60.15<sub>1.58</sub></td>
    <td align="center">19.49<sub>0.88</sub></td>
    <td align="center">2.27<sub>0.44</sub></td>
    <td align="center"><u>96.05</u><sub>0.34</sub></td>
    <td align="center"><u>67.83</u><sub>0.90</sub></td>
    <td align="center">21.93<sub>1.34</sub></td>
    <td align="center">2.35<sub>0.41</sub></td>
  </tr>

  <tr>
    <td align="center"><a href="https://github.com/weixi-feng/Structured-Diffusion-Guidance">Structured</a></td>
    <td align="center">90.40<sub>1.06</sub></td>
    <td align="center">48.64<sub>1.32</sub></td>
    <td align="center">10.71<sub>0.92</sub></td>
    <td align="center">0.68<sub>0.25</sub></td>
    <td align="center">89.25<sub>0.72</sub></td>
    <td align="center">53.05<sub>1.20</sub></td>
    <td align="center">15.76<sub>0.86</sub></td>
    <td align="center">1.74<sub>0.49</sub></td>
  </tr>

  <tr>
    <td align="center"><a href="https://github.com/yuval-alaluf/Attend-and-Excite">Attn-Exct</a></td>
    <td align="center"><u>93.64</u><sub>0.76</sub></td>
    <td align="center"><u>65.10</u><sub>1.24</sub></td>
    <td align="center"><u>28.01</u><sub>0.90</sub></td>
    <td align="center"><b>6.01</b><sub>0.61</sub></td>
    <td align="center">91.74<sub>0.49</sub></td>
    <td align="center">62.51<sub>0.94</sub></td>
    <td align="center"><u>26.12</u><sub>0.78</sub></td>
    <td align="center"><u>5.89</u><sub>0.40</sub></td>
  </tr>

  <tr>
    <td align="center"><a href="https://github.com/mlpc-ucsd/TokenCompose">Ours</a></td>
    <td align="center"><b>98.08</b><sub>0.40</sub></td>
    <td align="center"><b>76.16</b><sub>1.04</sub></td>
    <td align="center"><b>28.81</b><sub>0.95</sub></td>
    <td align="center"><u>3.28</u><sub>0.48</sub></td>
    <td align="center"><b>97.75</b><sub>0.34</sub></td>
    <td align="center"><b>76.93</b><sub>1.09</sub></td>
    <td align="center"><b>33.92</b><sub>1.47</sub></td>
    <td align="center"><b>6.21</b><sub>0.62</sub></td>
  </tr>

</table>

## 💻 Environment Setup

For those who want to use our codebase to **train your own diffusion models with grounding objectives**, follow the below instructions:

```bash
conda create -n TokenCompose python=3.8.5
conda activate TokenCompose
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

We have verified the environment setup using this specific package versions, but we expect that it will also work for newer versions too!

## 🛠️ Dataset Setup

If you want to use your own data, please refer to [preprocess_data](preprocess_data/readme.md) for details.

If you want to use our training data as examples or for research purposes, please follow the below instructions:

### 1. Setup the COCO Image Data

```bash
cd train/data
# download COCO train2017
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip
bash coco_data_setup.sh
```

After this step, you should have the following structure under the `train/data`  directory:

```
train/data/
    coco_gsam_img/
        train/
            000000000142.jpg
            000000000370.jpg
            ...
```


### 2. Setup Token-wise Grounded Segmentation Maps

Download COCO segmentation data from [Google Drive](https://drive.google.com/file/d/16uoQpfZ0O-NW92HuaCaFU8K4cGHHbv4R/view?usp=drive_link) and put it under `train/data` directory.

After this step, you should have the following structure under the `train/data` directory:

```
train/data/
    coco_gsam_img/
        train/
            000000000142.jpg
            000000000370.jpg
            ...
    coco_gsam_seg.tar
```

Then, run the following command to unzip the segmentation data:

```bash
cd train/data
tar -xvf coco_gsam_seg.tar
rm coco_gsam_seg.tar
```

After the setup, you should have the following structure under the `train/data` directory:

```
train/data/
    coco_gsam_img/
        train/
            000000000142.jpg
            000000000370.jpg
            ...
    coco_gsam_seg/
        000000000142/
            mask_000000000142_bananas.png
            mask_000000000142_bread.png
            ...
        000000000370/
            mask_000000000370_bananas.png
            mask_000000000370_bread.png
            ...
        ...
```

## 📈 Training 
We use wandb to log some curves and visualizations. Login to wandb before running the scripts.
```bash
wandb login
```
Then, to run TokenCompose, use the following command:

```bash
cd train
bash scripts/train.sh
```

The results will be saved under `train/results` directory.

## 🏷️ License

This repository is released under the [Attribution-NonCommercial 4.0 International license](LICENSE). 

## 🙏 Acknowledgement

Our code is built upon [diffusers](https://github.com/huggingface/diffusers), [prompt-to-prompt](https://github.com/google/prompt-to-prompt), [VISOR](https://github.com/microsoft/VISOR), [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything), and [CLIP](https://github.com/openai/CLIP). We thank all these authors for their nicely open sourced code and their great contributions to the community.

## 📝 Citation

If you find our work useful, please consider citing:
```
Release soon!
```
