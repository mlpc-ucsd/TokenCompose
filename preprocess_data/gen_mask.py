'''
Following code is adapted from
https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam_demo.py
'''

import argparse
import os
import copy
import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def construct_prompt(attn_lists):
    word_list = [attn_list[0] for attn_list in attn_lists]
    prompt = ' . '.join(word_list)
    return prompt, word_list

def aggregate_masks(masks, pred_phrases):
    phrased_masks = {}
    for mask, pred_phrase in zip(masks, pred_phrases):
        pred_phrase = pred_phrase.split('(')[0] # remove confidence postfix
        if pred_phrase not in phrased_masks:
            phrased_masks[pred_phrase] = mask
        else:
            phrased_masks[pred_phrase] += mask
    for pred_phrase in phrased_masks:
        # binary segmentation map over union
        phrased_masks[pred_phrase] = phrased_masks[pred_phrase] > 0 
    return phrased_masks

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument(
        "--input_metadata", type=str, default=None, help="path to metadata file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--output_jsonl", type=str, required=True, help="output path and name for jsonl")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    input_metadata = args.input_metadata
    output_dir = args.output_dir
    output_jsonl = args.output_jsonl
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    
    # load groudning model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    output_list = []

    with open(input_metadata, 'r') as json_file:
        json_list = list(json_file)

    counter = 0
    detected_objects = 0
    mismatched_objects = 0

    for i in tqdm(range(len(json_list))):
        json_str = json_list[i]
        counter += 1

        d = json.loads(json_str)
        image_path = d['file_name']
        text_prompt, word_list = construct_prompt(d['attn_list'])
        img_id = os.path.basename(image_path).split('.')[0]
        cur_sub_dir = os.path.join("seg", img_id)
        image_out_dir = os.path.join(output_dir, cur_sub_dir)
        os.makedirs(image_out_dir, exist_ok=True)

        # load image
        image_pil, image = load_image(image_path)

        # visualize raw image
        image_pil.save(os.path.join(image_out_dir, 'src.jpg'))

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
        try:
            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False,
            )
        except:
            print(f"Cannot predict {d['file_name']}")
            print(f"pred_phrases: {pred_phrases}")
            continue

        phrased_masks = aggregate_masks(masks, pred_phrases)
        for pred_phrase in phrased_masks:
            attn_list_phrase = None
            if pred_phrase not in word_list:
                # rule out case sensitivity
                if pred_phrase.upper() in word_list:
                    attn_list_phrase = pred_phrase.upper()
                elif pred_phrase.lower() in word_list:
                    attn_list_phrase = pred_phrase.lower()
                elif pred_phrase.replace(' - ', '-') in word_list:
                    attn_list_phrase = pred_phrase.replace(' - ', '-')
                else:
                    for word in word_list:
                        # if pred_phrase contains a word (e.g., pp: top horse; w: horse), then use the word (e.g., horse) as the phrase
                        # if word contains a pred_phrase (e.g., pp: horse; w: top horse), then use the pred_phrase (e.g., horse) as the phrase
                        if word.lower() in pred_phrase.lower().split(' ') or pred_phrase.lower() in word.lower().split(' '):
                            attn_list_phrase = word
                            break
                    for word in word_list:
                        # phrase to phrase mismatch (e.g., tank birthday vs tank birthday cake)
                        if word.lower() in pred_phrase.lower() or pred_phrase.lower() in word.lower():
                            attn_list_phrase = word
                            break

                    if attn_list_phrase is None:
                        # word list: ['a red and blue dump truck', 'ardula', 'a city street']
                        # pred phrase: 'a red dump truck'
                        pred_phrase_list = pred_phrase.split(' ')
                        for word in word_list:
                            all_token_in_word = True
                            for pred_token in pred_phrase_list:
                                if pred_token.lower() not in word.lower():
                                    all_token_in_word = False
                                    break
                            if all_token_in_word:
                                attn_list_phrase = word
                                break
            else:
                attn_list_phrase = pred_phrase

            if attn_list_phrase is None:
                print(f"Cannot find {pred_phrase} in {word_list}")
                continue

            mask = phrased_masks[pred_phrase].squeeze(0).cpu().numpy()

            mask_name = attn_list_phrase.replace(' ', '_')
            file_comps = [d['file_name'].split('/')[-1].split('.')[0]]
            file_comps.append(mask_name)
            file_comps.insert(0, 'mask')
            file_name = '_'.join(file_comps) + '.png'
            file_name = file_name.replace("/", "_") # safety check, in case os error
            save_path = os.path.join(image_out_dir, file_name)
            plt.imsave(save_path, mask, cmap=cm.gray)

            path_for_d = save_path.split('/')[-3:]
            path_for_d = '/'.join(path_for_d)
            for i in range(len(d['attn_list'])):
                if d['attn_list'][i][0] == attn_list_phrase:
                    # d['attn_list'][i][1] = path_for_d
                    d['attn_list'][i][1] = os.path.join(cur_sub_dir, file_name)
                    break

        d["file_name"] = os.path.basename(d["file_name"])
        output_list.append(d)

    with open(output_jsonl, 'w') as fw:
        for d in output_list:
            fw.write(json.dumps(d))
            fw.write("\n")

    