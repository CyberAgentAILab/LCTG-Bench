import os
import argparse
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

from utils.model_info_getter import get_suffix


load_dotenv("../.env")

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

def get_ctg_scores(model_path: str, f_name: str, iter_num: int, task: str) -> Tuple[float, float, float, float]:
    '''

    :param model_path: モデルのパス
    :param f_name: ファイル名
    :param iter_num: 実行回数
    :param task: タスク(summary, ad_text, pros_and_cons)
    :return: format, char_count, keyword, prohibited_wordのスコア
    '''
    output_list = []
    for i in range(iter_num):
        wip = []
        result_path = os.path.join("../output", task, model_path, "score", f_name+"_"+str(i)+".txt")
        with open(result_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Format" in line:
                    wip.append(float(line.split(": ")[1]))
                elif "Char count" in line:
                    wip.append(float(line.split(": ")[1]))
                elif "Keyword" in line:
                    wip.append(float(line.split(": ")[1]))
                elif "Prohibited word" in line:
                    wip.append(float(line.split(": ")[1]))
                else:
                    print(line)
                    raise RuntimeError("Invalid line")
        if len(wip) != 4:
            raise RuntimeError("Invalid result file")
        output_list.append(wip)

    format_score = 0
    char_count_score = 0
    keyword_score = 0
    prohibited_word_score = 0

    for output in output_list:
        format_score += output[0]
        char_count_score += output[1]
        keyword_score += output[2]
        prohibited_word_score += output[3]

    return (format_score/iter_num, char_count_score/iter_num, keyword_score/iter_num, prohibited_word_score/iter_num)

def get_quality_scores(model_path: str, f_name: str, task: str) -> Tuple[float, float, float, float]:
    result_path = os.path.join("../output", task, model_path, "score", f_name)
    output_list = list()
    with open(result_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "All-Acc" in line:
                output_list.append(float(line.split(": ")[1]))
        if len(output_list) != 4:
            raise RuntimeError("Invalid result file")
    return tuple(output_list)


def main(args):
    model_path_list = _get_model_paths()
    model_path_list = [get_suffix(m) for m in model_path_list]
    for task in ["summary", "ad_text", "pros_and_cons"]:
        if not os.path.isdir(f"../output/{task}"):
            continue
        scores_list = list()
        model_list = list()
        for model_path in model_path_list:
            if not os.path.isdir(os.path.join("../output", task, model_path, "score")):
                continue
            ctg_scores = get_ctg_scores(model_path, args.result_ctg_file, args.iter_num, task)
            quality_scores = get_quality_scores(model_path, args.result_quality_file, task)

            scores = list()
            for c, q in zip(ctg_scores, quality_scores):
                scores.append(c)
                scores.append(q)
            scores_list.append(scores)
            model_list.append(model_path)

        df = pd.DataFrame(data=scores_list, index=model_list, columns=["Format-ctg", "Format-qual", "C-count-ctg", "C-count-qual", "Keyword-ctg", "Keyword-qual", "P-word-ctg", "P-word-qual"])
        df["Avg-ctg"] = df.apply(lambda row: (row["Format-ctg"] + row["C-count-ctg"] + row["Keyword-ctg"] + row["P-word-ctg"]) / 4, axis=1)
        df["Avg-qual"] = df.apply(lambda row: (row["Format-qual"] + row["C-count-qual"] + row["Keyword-qual"] + row["P-word-qual"]) / 4, axis=1)
        for col in list(df.columns):
            df[col] = df[col].map(lambda f: "{:.3f}".format(f))
        df.to_csv(f"../output/{task}/all_scores.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_ctg_file", type=str, default=f"generated_result_v{os.environ['VERSION']}_ctg_score", help="ctgの結果のファイル名(拡張子なし)")
    parser.add_argument("--result_quality_file", type=str, default=f"generated_result_v{os.environ['VERSION']}_quality_score.txt", help="qualityの結果のファイル名")
    parser.add_argument("--iter_num", type=int, default=3, help="生成回数")
    args = parser.parse_args()
    main(args)
