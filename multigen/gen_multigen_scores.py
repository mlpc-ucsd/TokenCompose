'''
Following Code is adapted from
https://github.com/microsoft/VISOR/blob/main/visor.py
'''

import json 
import argparse
from tabulate import tabulate
import numpy as np 
from glob import glob

NUM_TO_PERCENT = 100

def get_multigen(results, threshold, text_data, num_ins, img_per_prompt):
	mg_n = [0] * num_ins # num_ins
	mg_std = np.zeros((img_per_prompt, num_ins), dtype=np.float32) # num_ins, img_per_prompt

	for img_id, rr in results.items():
		uniq_id = int(img_id.split("_")[0])
		img_per_prompt_id = int(img_id.split("_")[1])
		ann = text_data[uniq_id]

		obj_list = [ann[f'obj_{i+1}_attributes'][0] for i in range(num_ins)]
		origin_detected = rr["classes"]
		origin_scores = rr["scores"]

		# filter by detection threshold		
		assert len(origin_detected) == len(origin_scores), \
			"length of detected and scores should be same"
		
		detected = [origin_detected[i] for i in range(len(origin_scores)) if origin_scores[i] > threshold]
		generated_cnt = len(list(set(detected) & set(obj_list))) # intersection of detected and obj_list

		for k in range(generated_cnt):
			mg_n[k] += 1
			mg_std[img_per_prompt_id][k] += 1

	mg_std = ((NUM_TO_PERCENT * mg_std) / (len(results) / img_per_prompt))
	mg_std = np.std(mg_std, axis=0)		
	
	# compute average
	mg_n = [NUM_TO_PERCENT * mg / len(results) for mg in mg_n]
	return mg_n, mg_std


if __name__ == "__main__":
	parser = argparse.ArgumentParser() 
	parser.add_argument("--result_path", required=True)
	parser.add_argument("--name", required=True)
	parser.add_argument("--threshold", type=float, default=0.1)
	parser.add_argument("--text_file_path", type=str, required=True)
	parser.add_argument("--num_ins", type=int, default=5)
	parser.add_argument("--img_per_prompt", type=int, default=10)

	args = parser.parse_args() 

	print("Reading Data ...")
	with open(args.text_file_path, 'r') as f:
		text_data = json.load(f)
	print("Processing ...")

	# actually result path is result dir
	results_path = args.result_path
	results_files = glob(results_path + "/*.json")
	results = {}

	for results_file in results_files:
		with open(results_file, "r") as f:
			temp = json.load(f)
			results.update(temp)

	mg_n, mg_std = get_multigen(
		results=results,
		threshold=args.threshold,
		text_data=text_data,
		num_ins=args.num_ins,
		img_per_prompt=args.img_per_prompt
	)

	mg_data = [args.name] + [f'{mg_n[i]:.2f}({mg_std[i]:.2f})' for i in range(args.num_ins)] + [str(len(results)), str(args.threshold)]

	headers = ['Model'] + [f'MG_{i+1}' for i in range(args.num_ins)] + ['Num_Imgs', "det_threshold"]
	
	print(
		tabulate(
			[mg_data], 
			headers=headers,
			)
		)
