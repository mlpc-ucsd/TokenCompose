
## Preprocess Data Pipeline

### Overview

Suppose you have `input.json` with following format:
```
[
    {
        "img_path": "/path/to/image1.jpg",
        "caption" : ["caption1", "caption2", ...]
    },
    ...
]
```

We provide an overview of this pipeline in plain words given the input json:
1. the pipeline will choose the best caption for each image using [CLIP](https://github.com/openai/CLIP). 
2. the pipeline will use [flair](https://github.com/flairNLP/flair) to extract all nouns from the selected caption.
3. the pipeline will use [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to segment the image based on the extracted noun words, and generate token-level correspondence in a json file.

### Setup Environment

#### 1.  Clone Grounded-Segment-Anything Repository

```bash
cd preprocess_data
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
```

#### 2. Setup Environment
```bash
conda create -n preprocess_data python=3.8.18
conda activate preprocess_data
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```
Then following the instructions in [Grounded-Segment-Anything repo](https://github.com/IDEA-Research/Grounded-Segment-Anything) to install the dependencies

After that, install CLIP and flair
```bash
pip install git+https://github.com/openai/CLIP.git
pip install flair
```

#### 3. Copy Necessary Subfolders
```bash
cp -r Grounded-Segment-Anything/segment_anything .
cp -r Grounded-Segment-Anything/GroundingDINO .
```

#### 4. Download Necessary Checkpoints

```bash
cd model_ckpt
# download the pretrained groundingdino-swin-tiny model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
Then, download `sam_hq_vit_h.pth` from [here](https://github.com/SysCV/sam-hq#model-checkpoints)


You should have the following files in the model_ckpt folder
```
model_ckpt/
    groundingdino_swint_ogc.pth
    sam_hq_vit_h.pth
```

## Run the pipeline

See `run_pipeline.sh` for an example of how to run the pipeline

You should change the following variables in `run_pipeline.sh` to your own path
```bash
INPUT_JSON_PATH=/path/to/input_json.json
OUTPUT_JSON_PATH=/path/to/output_json.json
OUTPUT_DIR=/path/to/segmentation_output_dir
```

Then, run the pipeline
```bash
bash run_pipeline.sh
```

You will get the segmentation maps in `OUTPUT_DIR` and the output json in `OUTPUT_JSON_PATH`

The segmentation maps will have the following format
```
OUTPUT_DIR/
    seg/
        image1/
            mask_image1_noun1.png
            mask_image1_noun2.png
            ...
        image2/
            mask_image2_noun1.png
            mask_image2_noun2.png
            ...
        ...
```

## Tokenizer Check

Due to typos in image captions (**be careful as this is very common**), there may have some words that can not be processed correctly by tokenizer.

We highly recommand to dry run your training pipeline first to check if there are any data issues or you may want to implement a fault tolerant training pipeline. If you decide to manually fix the typos from your custom captions, we provide a sample script below:

You should change the following variable in `tokenizer_check.sh`
```
INPUT_JOSN_PATH=/path/to/output.json
```

Then, run the tokenizer check
```bash
bash tokenizer_check.sh
```

And manually fix all typo. Then you can use the processed data as your training data.

