import os
import json
from tqdm import tqdm

input_json_path = "coco_gsam_img/train/metadata.jsonl"

src_dir = "train2017"
tgt_dir = "coco_gsam_img/train"

input_json_data = []
with open(input_json_path, "r") as f:
    for line in f:
        input_json_data.append(json.loads(line))

for json_data in tqdm(input_json_data):
    file_name = json_data["file_name"]
    src_path = os.path.join(src_dir, file_name)
    tgt_path = os.path.join(tgt_dir, file_name)

    command = f"cp {src_path} {tgt_path}"
    os.system(command)
