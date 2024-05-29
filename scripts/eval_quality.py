import argparse
import os
import ast

from openai import OpenAI
import pandas as pd
import random
from tqdm import tqdm

from dotenv import load_dotenv

from utils.model_info_getter import get_suffix
from utils.prompt import get_quality_check_prompt, get_quality_check_prompt_bool

load_dotenv("../.env")
EVALUATOR_MODEL = "gpt-4-1106-preview"


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

def get_directories(path):
    directories = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir():
                directories.append(entry.name)
    return directories

def _delete_blank(result: str) -> str:
    result = result.replace(" ", "")
    result = result.replace("　", "")
    result = result.replace("\n", "")
    result = result.replace("\t", "")
    return result

def check_generated_result(client: OpenAI, task: str, generated_result: str, base_text: str = None, is_bool: bool = False):
    if is_bool:
        user_content = get_quality_check_prompt_bool(task, generated_result, base_text)
    else:
        user_content = get_quality_check_prompt(task, generated_result, base_text)

    completion = client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[
            {
                "role": "system",
                "content": "あなたは指示に忠実に従うAIアシスタントです。ユーザーの指示に従って下さい。"
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )
    result = completion.choices[0].message.content
    result = _delete_blank(result)
    return result


def _str_to_bool(str_list) -> bool:
    if isinstance(str_list, list):
        return str_list
    bool_list = [bool(s) for s in ast.literal_eval(str_list)]
    return bool_list


def main(args):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    if (args.model_num is not None) and (args.model_start_index is not None):
        model_path_list = _get_model_paths()[args.model_start_index:args.model_start_index + args.model_num]
    elif (args.model_num is None) and (args.model_start_index is None):
        model_path_list = _get_model_paths()
    else:
        raise RuntimeError("Modelの引数の与え方がまずってます。")
    model_path_list = [get_suffix(m) for m in model_path_list]

    print("==============Quality Check Step. ==================")
    for model_name in model_path_list:
        print("==============Model is {}==================".format(model_name))
        result_path = os.path.join(args.result_path, args.task, model_name, "generated_result", args.result_file)
        df = pd.read_json(result_path, orient="records", lines=True)
        result_dict = dict()
        result_dict["generated_text_id"] = df["generated_text_id"].tolist()
        result_dict["prompt_id"] = df["prompt_id"].tolist()
        result_dict["model"] = model_name
        ctg_cols = [col for col in df.columns if "wo_hf" in col]
        for ctg_col in ctg_cols:
            print(f"Perspective of controllability: {ctg_col}")
            ctg = ctg_col.replace("_result", "")
            check_result_list = list()
            for base_text, generated_text_list in tqdm(zip(df["base_text"].tolist(), df[ctg_col].tolist()), total=len(df)):
                check_result = list()
                for generated_text in generated_text_list:
                    cr = check_generated_result(client, args.task, generated_text, base_text)
                    if (cr == "適切") or (cr == "不適切"):
                        label_map = {"適切": True, "不適切": False}
                        check_result.append(label_map[cr])
                    else:
                        cr = check_generated_result(client, args.task, generated_text, base_text, is_bool=True)
                        if cr == "True" or cr == "False":
                            label_map = {"True": True, "False": False}
                            check_result.append(label_map[cr])
                        else:
                            check_result.append(cr)
                check_result_list.append(check_result)
            result_dict[f"{ctg}"] = check_result_list
        df_result = pd.DataFrame(result_dict)
        output_path = os.path.join(args.output_path, args.task, model_name, "score")
        os.makedirs(output_path, exist_ok=True)
        df_result.to_csv(f"{output_path}/generated_result_v{os.environ['VERSION']}_quality_score.csv", index=False)
        
        with open(f"{output_path}/generated_result_v{os.environ['VERSION']}_quality_score.txt", "w") as f:
            f.write("Model: " + model_name + "\n")
            f.write("Data Num: " + str(len(df_result)) + "\n")
            for ctg_col in df_result.columns:
                if "wo_hf" not in ctg_col:
                    continue
                f.write(f"【{ctg_col}】" + "\n")

                i0_list = [_str_to_bool(i)[0] for i in df_result[ctg_col].tolist()]
                i0_list = [i for i in i0_list if isinstance(i, bool)]
                acc0 = i0_list.count(True) / len(i0_list)
                f.write("iter0-Acc: {:.3f} ({} / {})\n".format(acc0, i0_list.count(True), len(i0_list)))

                i1_list = [_str_to_bool(i)[1] for i in df_result[ctg_col].tolist()]
                i1_list = [i for i in i1_list if isinstance(i, bool)]
                acc1 = i1_list.count(True) / len(i1_list)
                f.write("iter1-Acc: {:.3f} ({} / {})\n".format(acc1, i1_list.count(True), len(i1_list)))

                i2_list = [_str_to_bool(i)[2] for i in df_result[ctg_col].tolist()]
                i2_list = [i for i in i2_list if isinstance(i, bool)]
                acc2 = i2_list.count(True) / len(i2_list)
                f.write("iter2-Acc: {:.3f} ({} / {})\n".format(acc2, i2_list.count(True), len(i2_list)))

                f.write("All-Acc: {:.3f}\n".format((acc0 + acc1 + acc2) / 3))
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ad_text", help="タスク名")
    parser.add_argument("--result_path", default=f"../output", help="生成結果のファイルの置いてあるディレクトリ")
    parser.add_argument("--result_file", default=f"generated_result_v{os.environ['VERSION']}_wo_hf.jsonl", help="タスク名")
    parser.add_argument("--output_path", default=f"../output", help="出力先のパス")
    parser.add_argument("--model_num", type=int, help="検証するモデルの数")
    parser.add_argument("--model_start_index", type=int, help="検証するモデルのインデックスの開始位置")
    args = parser.parse_args()
    main(args)
