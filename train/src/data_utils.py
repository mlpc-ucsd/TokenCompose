'''
The folloing code is partly adapted from
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
'''
from PIL import Image
import os

class DatasetPreprocess:
    def __init__(self, caption_column, image_column, train_transforms, attn_transforms, tokenizer, train_data_dir, segment_dir_origin_path="seg", segment_dir_relative_path="../coco_gsam_seg"):
        self.caption_column = caption_column
        self.image_column = image_column

        self.train_transforms = train_transforms
        self.attn_transforms = attn_transforms

        self.tokenizer = tokenizer

        self.train_data_dir = train_data_dir
        self.segment_dir_origin_path = segment_dir_origin_path
        self.segment_dir_relative_path = segment_dir_relative_path

    def tokenize_captions(self, examples):
        captions = []
        for caption in examples[self.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            else:
                raise ValueError(
                    f"Caption column `{self.caption_column}` should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions, max_length = self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def data_preprocess_train(self, examples):

        images = [image.convert("RGB") for image in examples[self.image_column]]
        
        examples["pixel_values"] = [self.train_transforms(image) for image in images]

        # process text
        examples["input_ids"] = self.tokenize_captions(examples)

        # read in attn gt, for hard attn map
        examples["postprocess_seg_ls"] = []
        for i in range(len(examples["attn_list"])):
            attn_list = examples["attn_list"][i]
            postprocess_attn_list = []
            for item in attn_list:
                category = item[0]
                attn_path = item[1]

                if attn_path is None:
                    # ignore if path is None
                    continue

                if not os.path.exists(attn_path):
                    attn_path = attn_path.replace(self.segment_dir_origin_path, self.segment_dir_relative_path)
                    attn_path = os.path.join(self.train_data_dir, attn_path)
                    if not os.path.exists(attn_path):
                        raise ValueError(f"attn path {attn_path} does not exist")

                attn_gt = Image.open(attn_path)
                attn_gt = self.attn_transforms(attn_gt)

                if attn_gt.shape[0] == 4:
                    # 4 * 512 * 512 -> 1 * 512 * 512
                    attn_gt = attn_gt[0].unsqueeze(0)

                # normalize to [0, 1]
                if attn_gt.max() > 0:
                    attn_gt = attn_gt / attn_gt.max()

                postprocess_attn_list.append([
                    category,
                    attn_gt
                ])

            examples["postprocess_seg_ls"].append(postprocess_attn_list)

        del examples["image"]
        del examples["attn_list"]

        return examples

    def preprocess(self, input_dataset):
        return input_dataset.with_transform(self.data_preprocess_train)
