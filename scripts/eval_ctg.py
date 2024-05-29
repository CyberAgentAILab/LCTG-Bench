import logging
import os
import argparse
import json
from typing import List

import pandas as pd
from dotenv import load_dotenv

from utils.model_info_getter import get_suffix

load_dotenv("../.env")
REF_INDEX = 0

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

def _delete_blank(s: str) -> str:
    s = s.replace(" ", "")
    s = s.replace("　", "")
    s = s.replace("\n", "")
    s = s.replace("\t", "")
    return s


def _is_valid_format(s: str, s_wo_hf) -> bool:
    if s_wo_hf == "Invalid Error":
        return False

    s = _delete_blank(s)
    s_wo_hf = _delete_blank(s_wo_hf)
    if len(s_wo_hf) > 10:
        if (s[:10] == s_wo_hf[:10]) and (s[-10:] == s_wo_hf[-10:]):
            return True
        else:
            return False


def _is_valid_keyword(s: str, keyword: str) -> bool:
    if s == "Invalid Error":
        return False
    return keyword in s


def _is_valid_prohibited_word(s: str, keyword: str) -> bool:
    if s == "Invalid Error":
        return False
    return keyword not in s


def _is_valid_char_count(s: str, char_count: List[int]) -> bool:
    if s == "Invalid Error":
        return False
    return (char_count[0] <= len(s)) and (len(s) <= char_count[1])


def get_format_is_valid(row: pd.Series) -> bool:
    return _is_valid_format(row["format_result"][REF_INDEX], row["format_result_wo_hf"][REF_INDEX])


def get_keyword_is_valid(row: pd.Series) -> bool:
    return _is_valid_keyword(row["keyword_result_wo_hf"][REF_INDEX], row["keyword_answer"])


def get_prohibited_word_is_valid(row: pd.Series) -> bool:
    return _is_valid_prohibited_word(row["prohibited_word_result_wo_hf"][REF_INDEX], row["prohibited_word_answer"])


def get_char_count_is_valid(row: pd.Series) -> bool:
    return _is_valid_char_count(row["char_count_result_wo_hf"][REF_INDEX], row["char_count_answer"])


def main(args):
    global REF_INDEX

    while REF_INDEX != 3:
        if (args.model_num is not None) and (args.model_start_index is not None):
            model_path_list = _get_model_paths()[args.model_start_index:args.model_start_index + args.model_num]
        elif (args.model_num is None) and (args.model_start_index is None):
            model_path_list = _get_model_paths()
        else:
            raise RuntimeError("wrong model path. Please check the arguments.")
        print("==============CTG Scoring Step. ==================")
        for model in model_path_list:
            print("==============Model is {}==================".format(model))
            model = get_suffix(model)
            master_path = os.path.join(args.master_path, args.task, args.master_file)
            master_df = pd.read_json(master_path, orient="records", lines=True)

            result_path = os.path.join(args.result_path, args.task, model, "generated_result", args.result_file)

            try:
                result_df = pd.read_json(result_path, orient="records", lines=True)
            except ValueError:
                logging.warning("{} is not found.".format(result_path))
                continue


            model_path = result_df["model"].tolist()[0]
            result_df = result_df[["prompt_id", "format_result", "format_result_wo_hf", "char_count_result", "char_count_result_wo_hf", "keyword_result", "keyword_result_wo_hf", "prohibited_word_result", "prohibited_word_result_wo_hf"]]
            df = pd.merge(master_df, result_df, on="prompt_id")
            df["is_valid_format"] = df.apply(get_format_is_valid, axis=1)
            df["is_valid_char_count"] = df.apply(get_char_count_is_valid, axis=1)
            df["is_valid_keyword"] = df.apply(get_keyword_is_valid, axis=1)
            df["is_valid_prohibited_word"] = df.apply(get_prohibited_word_is_valid, axis=1)

            score_list = list()
            score_list.append(df['is_valid_format'].sum() / len(df))
            score_list.append(df['is_valid_char_count'].sum() / len(df))
            score_list.append(df['is_valid_keyword'].sum() / len(df))
            score_list.append(df['is_valid_prohibited_word'].sum() / len(df))

            print(f"Model: {model_path}, Iteration: {REF_INDEX}")
            print("Format: {:.3f}".format(score_list[0]))
            print("Char count: {:.3f}".format(score_list[1]))
            print("Keyword: {:.3f}".format(score_list[2]))
            print("Prohibited word: {:.3f}".format(score_list[3]))
            print("========================")

            output_path_analysis = os.path.join(args.output_path, args.task, model, "analysis")
            os.makedirs(output_path_analysis, exist_ok=True)
            if "title" in list(df.columns):
                tmp = df[["prompt_id", "title", "base_text", "char_count", "char_count_result", "char_count_result_wo_hf", "is_valid_char_count", "keyword", "keyword_result", "keyword_result_wo_hf", "is_valid_keyword", "prohibited_word", "prohibited_word_result", "prohibited_word_result_wo_hf", "is_valid_prohibited_word", "format", "format_result", "format_result_wo_hf", "is_valid_format"]]
                save_df = tmp.copy()

            else:
                tmp = df[
                    ["prompt_id", "base_text", "char_count", "char_count_result", "char_count_result_wo_hf",
                     "is_valid_char_count", "keyword", "keyword_result", "keyword_result_wo_hf", "is_valid_keyword",
                     "prohibited_word", "prohibited_word_result", "prohibited_word_result_wo_hf",
                     "is_valid_prohibited_word", "format", "format_result", "format_result_wo_hf", "is_valid_format"]]
                save_df = tmp.copy()
            save_df.loc[:, "format_result"] = save_df["format_result"].map(lambda x: x[REF_INDEX])
            save_df.loc[:, "format_result_wo_hf"] = save_df["format_result_wo_hf"].map(lambda x: x[REF_INDEX])
            save_df.loc[:, "char_count_result"] = save_df["char_count_result"].map(lambda x: x[REF_INDEX])
            save_df.loc[:, "char_count_result_wo_hf"] = save_df["char_count_result_wo_hf"].map(lambda x: x[REF_INDEX])
            save_df.loc[:, "keyword_result"] = save_df["keyword_result"].map(lambda x: x[REF_INDEX])
            save_df.loc[:, "keyword_result_wo_hf"] = save_df["keyword_result_wo_hf"].map(lambda x: x[REF_INDEX])
            save_df.loc[:, "prohibited_word_result"] = save_df["prohibited_word_result"].map(lambda x: x[REF_INDEX])
            save_df.loc[:, "prohibited_word_result_wo_hf"] = save_df["prohibited_word_result_wo_hf"].map(lambda x: x[REF_INDEX])
            save_df.to_csv(f"{output_path_analysis}/generated_result_v{os.environ['VERSION']}_analysis_{REF_INDEX}.csv", index=False)
            save_df.to_csv(f"{output_path_analysis}/generated_result_v{os.environ['VERSION']}_analysis_{REF_INDEX}.tsv", sep="\t", index=False)
            save_df.to_json(f"{output_path_analysis}/generated_result_v{os.environ['VERSION']}_analysis_{REF_INDEX}.jsonl", orient="records", lines=True, force_ascii=False)

            output_path_score = os.path.join(args.output_path, args.task, model, "score")
            os.makedirs(output_path_score, exist_ok=True)
            with open(f"{output_path_score}/generated_result_v{os.environ['VERSION']}_ctg_score_{REF_INDEX}.txt", "w") as f:
                for prefix, score in zip(["Format", "Char count", "Keyword", "Prohibited word"], score_list):
                    f.write("{}: {:.3f}\n".format(prefix, score))
        REF_INDEX += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_path", default="../datasets", help="プロンプトのパス")
    parser.add_argument("--master_file", default=f"prompts_v{os.environ['VERSION']}.jsonl", help="プロンプトのファイル名")
    parser.add_argument("--task", default="ad_text", help="タスク名")
    parser.add_argument("--result_path", default=f"../output", help="生成結果のファイルの置いてあるディレクトリ")
    parser.add_argument("--result_file", default=f"generated_result_v{os.environ['VERSION']}_wo_hf.jsonl", help="タスク名")
    parser.add_argument("--output_path", default=f"../output", help="出力先のパス")
    parser.add_argument("--model_num", type=int, help="検証するモデルの数")
    parser.add_argument("--model_start_index", type=int, help="検証するモデルのインデックスの開始位置")
    args = parser.parse_args()
    main(args)
