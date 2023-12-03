'''
Following code is adapted from
https://github.com/openai/CLIP
https://github.com/flairNLP/flair
'''

import clip
from tqdm import tqdm
from PIL import Image
import torch
import json
import os
from flair.data import Sentence
from flair.models import SequenceTagger
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

# load tagger
tagger = SequenceTagger.load("flair/pos-english")

def get_nouns(tag_l):
    n_phrases = []
    for i in range(len(tag_l)):
        if i == 0 and 'NN' in tag_l[i]['entity']:
            n_phrases.append(tag_l[i]['word'])
        elif i > 0 and 'NN' in tag_l[i]['entity']:
            if len(n_phrases) == 0:
                n_phrases.append(tag_l[i]['word'])
            else:
                if 'NN' in tag_l[i - 1]['entity']:
                    if tag_l[i]['word'].startswith('##'):
                        n_phrases[-1] = n_phrases[-1] + tag_l[i]['word'].replace('##', '')
                    elif tag_l[i - 1]['word'].endswith('@@'):
                        n_phrases[-1] = n_phrases[-1].replace('@@', '') + tag_l[i]['word']
                    else:
                        n_phrases[-1] = n_phrases[-1] + ' ' + tag_l[i]['word']
                else:
                    n_phrases.append(tag_l[i]['word'])
    return n_phrases

def preprocess_img_captions(img_path, caption):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    if isinstance(caption, list):
        # use clip to find the best caption
        text = clip.tokenize(caption).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        best_caption_idx = probs.argmax()
        caption = caption[best_caption_idx]

    sentence = Sentence(caption)
    tagger.predict(sentence)

    l = []
    for i in range(len(sentence.tokens)):
        d_t = {
            'word': sentence.tokens[i].text,
            'entity': sentence.tokens[i].tag
        }
        l.append(d_t)
    attn_list_nouns = list(set(get_nouns(l)))
    attn_list = [[chunk, None] for chunk in attn_list_nouns]
    d = {
        "file_name": img_path,
        "text": caption,
        "attn_list": attn_list
    }
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", type=str, required=True)
    parser.add_argument("--output_json_path", type=str, required=True)
    args = parser.parse_args()

    input_json_path = args.input_json_path
    output_json_path = args.output_json_path
    
    input_json_data = []
    with open(input_json_path, 'r') as f:
        for line in f:
            input_json_data.append(json.loads(line.strip()))

    output_json_data = []

    for json_data in tqdm(input_json_data):

        img_path = json_data['img_path']
        caption = json_data['caption']

        d = preprocess_img_captions(img_path, caption)
        output_json_data.append(d)

    with open(output_json_path, 'w') as f:
        for d in output_json_data:
            f.write(json.dumps(d) + '\n')
