## MultiGen

### Overview

**MultiGen** is designed to be a challenging metric for multi-category instance composition. Specifically, given a set of distinct instance categories of size $N$, we randomly sample 5 categories (*e.g,* A, B, C, D, E), format them into a sentence (*i.e.,* A photo of A, B, C, D, and E.), and use them as the condition for a text-to-image diffusion model to generate the image.

### Run the pipeline

See `run_pipeline.sh` for an example of how to run the pipeline

You should change the following variables in `run_pipeline.sh` to your own path
```bash
# choose from ade20k_obj_comp_5_1k.json, coco_obj_comp_5_1k.json
TEXT_FILE_PATH=/path/to/text_file.json

# if you want to use a checkpoint from huggingface, set MODEL_NAME to huggingface model name
# For example, if you want to test Stable Diffusion 1.4, set MODEL_NAME=CompVis/stable-diffusion-v1-4
# If you want to test our model, choose MODEL_NAME from the following:
# mlpc-lab/TokenCompose_SD14_A, mlpc-lab/TokenCompose_SD14_B, mlpc-lab/TokenCompose_SD21_A, mlpc-lab/TokenCompose_SD21_B
MODEL_NAME=/model/to/test
```

Notice, current pipeline only support inference of diffusers model from huggingface. If you want to test other models, you need to modify `run_pipeline.sh` and `gen_image_dist.py` accordingly.

Then, run the pipeline

```bash
conda activate TokenCompose
bash run_pipeline.sh
```

### Output Format

The final output should have the following format
```
Model    MG_1         MG_2         MG_3         MG_4         MG_5         Num_Imgs    det_threshold
-------  -----------  -----------  -----------  -----------  -----------  ----------  ---------------
NAME     SR_1(STD_1)  SR_2(STD_2)  SR_3(STD_3)  SR_4(STD_4)  SR_5(STD_5)  NUM_IMGS    DET_THRESHOLD
```

where `SR` stands for `success rate`, `STD` stands for `standard deviation`.
