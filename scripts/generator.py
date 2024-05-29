import argparse
import os
import logging

from tqdm import tqdm
import torch
from dotenv import load_dotenv
import pandas as pd

from utils.model_info_getter import get_suffix, get_model_tokenizer, get_output_tokens


load_dotenv("../.env")
limitation_type_list = ["format", "char_count", "keyword", "prohibited_word"]


def get_model_paths():
    model_paths = []
    i = 0
    while True:
        key = f'MODEL_PATH_{i}'
        if key in os.environ:
            model_paths.append(os.environ[key])
            i += 1
        else:
            break
    return model_paths


def main(args):
    if (args.model_num is not None) and (args.model_start_index is not None):
        model_path_list = get_model_paths()[args.model_start_index:args.model_start_index + args.model_num]
    elif (args.model_num is None) and (args.model_start_index is None):
        model_path_list = get_model_paths()
    else:
        raise RuntimeError("Invalid Model Arguments.")

    master_path = os.path.join(args.master_path, args.task, args.master_file)
    master_df = pd.read_json(master_path, orient="records", lines=True)

    start_index = 0
    print("==============Inference Step. ==================")
    for model_path in model_path_list:
        print("==============Model is {}==================".format(model_path))
        all_model_list = list()
        all_prompt_id_list = list()
        all_base_text_list = list()
        all_char_count_list = list()
        all_keyword_list = list()
        all_prohibited_word_list = list()
        all_format_list = list()

        all_prompt_id_list.extend(master_df["prompt_id"].tolist())
        all_base_text_list.extend(master_df["base_text"].tolist())
        all_model_list.extend([model_path] * len(master_df))
        model, tokenizer = get_model_tokenizer(model_path)

        for lmt_type in limitation_type_list:
            print(f"Perspective of controllability: {lmt_type}")
            prompt_list = master_df[f"prompt_{lmt_type}"].tolist()
            output_tokens_list = list()
            use_system_prompt = args.use_system_prompt

            for prompt in tqdm(prompt_list):
                wip = list()
                for i in range(args.iter_num):
                    output_tokens = get_output_tokens(model_path, model, tokenizer, prompt, use_system_prompt)
                    wip.append(output_tokens)
                output_tokens_list.append(wip)

            if lmt_type == "char_count":
                all_char_count_list.extend(output_tokens_list)
            elif lmt_type == "keyword":
                all_keyword_list.extend(output_tokens_list)
            elif lmt_type == "prohibited_word":
                all_prohibited_word_list.extend(output_tokens_list)
            elif lmt_type == "format":
                all_format_list.extend(output_tokens_list)
            else:
                raise RuntimeError("lmt_type is invalid.")

        keys = list()
        values = list()
        keys.append("generated_text_id")
        values.append(list(range(start_index, start_index + len(all_prompt_id_list))))
        keys.append("model")
        values.append(all_model_list)
        keys.append("prompt_id")
        values.append(all_prompt_id_list)
        keys.append("base_text")
        values.append(all_base_text_list)

        if all_format_list != []:
            keys.append("format_result")
            values.append(all_format_list)
        if all_char_count_list != []:
            keys.append("char_count_result")
            values.append(all_char_count_list)
        if all_keyword_list != []:
            keys.append("keyword_result")
            values.append(all_keyword_list)
        if all_prohibited_word_list != []:
            keys.append("prohibited_word_result")
            values.append(all_prohibited_word_list)

        data = dict()
        for k, v in zip(keys, values):
            if k not in data.keys():
                data[k] = v

        df = pd.DataFrame(data)
        output_path = os.path.join(args.output_path, args.task, get_suffix(model_path), "generated_result")
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"generated_result_v{os.environ['VERSION']}.jsonl")
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        start_index += len(all_prompt_id_list)
        del model
        del tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_num", type=int, help="検証するモデル数")
    parser.add_argument("--model_start_index", type=int, help="モデルの開始インデックス")
    parser.add_argument("--master_path", default="../datasets", help="プロンプトのパス")
    parser.add_argument("--master_file", default=f"prompts_v{os.environ['VERSION']}.jsonl", help="プロンプトのファイル名")
    parser.add_argument("--output_path", default="../output", help="出力先のパス（基本触らない）")
    parser.add_argument("--task", default="ad_text", help="タスク名")
    parser.add_argument("--iter_num", type=int, default=3, help="実行回数")
    parser.add_argument('--use-system-prompt', dest='use_system_prompt', action='store_true')
    parser.add_argument('--no-use-system-prompt', dest='use_system_prompt', action='store_false')
    parser.set_defaults(use_system_prompt=True)
    args = parser.parse_args()
    main(args)
