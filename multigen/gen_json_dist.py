'''
Following code is adapted from
https://github.com/microsoft/VISOR/blob/main/text2image_realtime_evaluate.ipynb
'''

import os
import json
import torch
import argparse

from tqdm import tqdm
from PIL import Image
from accelerate import PartialState
from transformers import Owlv2Processor, Owlv2ForObjectDetection

def mg_by_index(uniq_id, image_dir, text_data, num_ins, img_per_prompt, 
                   threshold, model, processor, device):

    image_file_list = os.listdir(image_dir)
    image_file_postfix = ".png"

    if image_file_list[0].endswith(".jpg"):
        image_file_postfix = ".jpg"

    inses = [text_data[uniq_id][f"obj_{i+1}_attributes"][0] for i in range(num_ins)]
    texts = [["a photo of a {}".format(ins) for ins in inses]]

    results = {}
    for i in range(img_per_prompt):

        # read image here
        img_id = "{}_{}".format(uniq_id, i)
        impath = os.path.join(image_dir, f"{img_id}{image_file_postfix}")
        image = Image.open(impath)

        # detection here
        with torch.no_grad():
            inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            target_sizes = torch.Tensor([image.size[::-1]]).to(device)
            outs = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)

        scores, labels = outs[0]["scores"], outs[0]["labels"]
        det_scores, det_labels = [], []
        for score, label in zip(scores, labels):
            # we save detections with score > 0.01 
            # but will later use 0.1 for evaluation
            if score > threshold:
                det_scores.append(score.tolist())
                det_labels.append(inses[label.item()])

        results[img_id] = {
            "classes": det_labels, 
            "scores": det_scores,
        }

    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_json_path", type=str, required=True)
    parser.add_argument("--text_file_path", type=str, required=True)
    parser.add_argument("--num_ins", type=int, default=5)
    parser.add_argument("--img_per_prompt", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--detector", type=str, default="google/owlv2-large-patch14-ensemble")

    args = parser.parse_args()

    with open(args.text_file_path, 'r') as f:
        text_data = json.load(f)

    processor = Owlv2Processor.from_pretrained(args.detector)
    model = Owlv2ForObjectDetection.from_pretrained(args.detector)

    distributed_state = PartialState()
    device = distributed_state.device
    
    model = model.to(device)
    model.eval()

    json_name = f"{os.environ['RANK']}.json"
    output_json_name = os.path.join(args.output_json_path, json_name)

    index_list = list(range(len(text_data)))
    with distributed_state.split_between_processes(index_list) as data:
        results_dict = {}

        for index in tqdm(data):
            results = mg_by_index(uniq_id=index,
                        image_dir=args.image_dir,
                        text_data=text_data,
                        num_ins=args.num_ins,
                        img_per_prompt=args.img_per_prompt,
                        threshold=args.threshold,
                        model=model,
                        processor=processor,
                        device=device)
            
            for key, value in results.items():
                results_dict[key] = value

        with open(output_json_name, "w") as f:
            json.dump(results_dict, f, indent=4, separators=(",", ": "))
