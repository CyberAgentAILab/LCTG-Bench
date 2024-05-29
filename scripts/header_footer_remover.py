import argparse
import os
import time
from typing import List
import logging

import openai
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv

from utils.model_info_getter import get_suffix
from utils.prompt import get_header_footer_remover_prompt


load_dotenv("../.env")


FORMATTER_MODEL = "gpt-4-1106-preview"


def _get_generated_result_wo_header_footer(client, gr_list_list: List[List[str]], model, task: str) -> List[List[str]]:
    response = list()
    for gr_list in tqdm(gr_list_list):
        gr_wo_hf_list = list()
        for gr in gr_list:
            while True:
                try:
                    completion = client.chat.completions.create(
                        model=FORMATTER_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": "あなたは指示に忠実に従うAIアシスタントです。ユーザーの指示に従って下さい。"
                            },
                            {
                                "role": "user",
                                "content": get_header_footer_remover_prompt(task, gr)
                            }
                        ]
                    )
                    output = completion.choices[0].message.content
                    gr_wo_hf_list.append(output)
                    break
                except openai.RateLimitError:
                    print("RateLimitError: sleep 30 seconds.")
                    time.sleep(30)
                except openai.BadRequestError:
                    print("BadRequestError: sleep 30 seconds.")
                    time.sleep(30)
                    output = "Invalid Error"
                    gr_wo_hf_list.append(output)
                    break
        response.append(gr_wo_hf_list)
    return response


def _get_model_paths():
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
        model_path_list = _get_model_paths()[args.model_start_index:args.model_start_index + args.model_num]
    elif (args.model_num is None) and (args.model_start_index is None):
        model_path_list = _get_model_paths()
    else:
        raise RuntimeError("Modelの引数の与え方がまずってます。")

    print("==============Formatting Step. ==================")
    for model in model_path_list:
        print("=============Model is {}==================".format(model))
        model = get_suffix(model)
        result_file = os.path.join(args.result_path, args.task, model,"generated_result",  args.result_file)
        try:
            result_df = pd.read_json(result_file, orient="records", lines=True)
        except ValueError:
            raise RuntimeError("{} is not found.".format(result_file))

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        ctg_col_list = ["format_result", "char_count_result", "keyword_result", "prohibited_word_result"]
        output_path = os.path.join(args.output_path, args.task, model, "generated_result")
        for ctg_col in ctg_col_list:
            print("Perspective of controllability: {}".format(ctg_col))
            generation_result_wo_hf = _get_generated_result_wo_header_footer(client, result_df[ctg_col].tolist(), model, args.task)
            result_df[f"{ctg_col}_wo_hf"] = generation_result_wo_hf

            result_df.to_json(os.path.join(output_path, f"generated_result_v{os.environ['VERSION']}_wo_hf.jsonl"),
                              lines=True, orient="records", force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_num", type=int, help="検証するモデルの数")
    parser.add_argument("--model_start_index", type=int, help="検証するモデルのインデックスの開始位置")
    parser.add_argument("--result_path", default=f"../output", help="生成結果のファイルの置いてあるディレクトリ")
    parser.add_argument("--result_file", default=f"generated_result_v{os.environ['VERSION']}.jsonl", help="タスク名")
    parser.add_argument("--task", default="ad_text", help="タスク名")
    parser.add_argument("--output_path", default="../output", help="出力先のパス")
    args = parser.parse_args()
    main(args)
