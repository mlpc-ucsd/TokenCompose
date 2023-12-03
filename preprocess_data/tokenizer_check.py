import json
from tqdm import tqdm
from transformers import CLIPTokenizer
import argparse


# get token index in text
def get_word_idx(text: str, tgt_word, tokenizer):

    tgt_word = tgt_word.lower()

    # ignore the first and last token
    encoded_text = tokenizer.encode(text)[1:-1]
    encoded_tgt_word = tokenizer.encode(tgt_word)[1:-1]

    # find the idx of target word in text
    first_token_idx = -1
    for i in range(len(encoded_text)):
        if encoded_text[i] == encoded_tgt_word[0]:

            if len(encoded_text) > 0:
                # check the following 
                following_match = True
                for j in range(1, len(encoded_tgt_word)):
                    if encoded_text[i + j] != encoded_tgt_word[j]:
                        following_match = False
                if not following_match:
                    continue
            # for a single encoded idx, just take it
            first_token_idx = i

            break

    assert first_token_idx != -1, "word not in text"

    # add 1 for sot token
    tgt_word_tokens_idx_ls = [i + 1 + first_token_idx for i in range(len(encoded_tgt_word))]

    # sanity check
    encoded_text = tokenizer.encode(text)

    decoded_token_ls = []

    for word_idx in tgt_word_tokens_idx_ls:
        text_decode = tokenizer.decode([encoded_text[word_idx]]).strip("#")
        decoded_token_ls.append(text_decode)

    decoded_tgt_word = "".join(decoded_token_ls)
    
    tgt_word_ls = tgt_word.split(" ")
    striped_tgt_word = "".join(tgt_word_ls).strip("#")

    assert decoded_tgt_word == striped_tgt_word, "decode_text != striped_tar_wd"

    return tgt_word_tokens_idx_ls


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", type=str, required=True)
    args = parser.parse_args()

    model_name ="CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name, subfolder="tokenizer"
    )

    input_json_path = args.input_json_path

    input_json_data = []
    with open(input_json_path, 'r') as f:
        for line in f:
            input_json_data.append(json.loads(line.strip()))

    for json_data in tqdm(input_json_data):
        file_name = json_data['file_name']
        text = json_data['text']
        attn_list = json_data['attn_list']

        for words, _ in attn_list:
            word_index = get_word_idx(text, words, tokenizer)
        
    






