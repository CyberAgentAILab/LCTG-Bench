from typing import List

import pandas as pd


def _get_prompt_list(base_text_list: List[str], limitations: List[str], task: str) -> List[str]:
    prompt_list = list()
    if task == "summary":
        for lmt, base_text in zip(limitations, base_text_list):
            prompt = f"""以下の条件で与えられた文章を要約して出力してください。
[条件]
{lmt}
[文章]
{base_text}
"""
            prompt_list.append(prompt)
    elif task == "ad_text":
        for lmt, base_text in zip(limitations, base_text_list):
            prompt = f"""以下の[文章]で与えられた説明文に対する広告文のタイトルを、[条件]に従って1つ作成してください。
[条件]
{lmt}
[文章]
{base_text}
"""
            prompt_list.append(prompt)
    elif task == "pros_and_cons":
        for lmt, base_text in zip(limitations, base_text_list):
            prompt = f"""{base_text}メリットとデメリットを以下の条件に従って文章で回答してください。
[条件]
{lmt}
"""
            prompt_list.append(prompt)
    return prompt_list


def get_prompt_list(lmt_type: str, master_df: pd.DataFrame, task: str) -> List[str]:
    if lmt_type == "char_num":
        limitations = master_df["char_count"].tolist()
        limitations = [f"{lmt}" for lmt in limitations]
    elif lmt_type == "keyword_use":
        limitations = master_df["keyword"].tolist()
        limitations = [f"{lmt}" for lmt in limitations]
    elif lmt_type == "prohibited_word":
        limitations = master_df["prohibited_word"].tolist()
        limitations = [f"{lmt}" for lmt in limitations]
    elif lmt_type == "format":
        limitations = master_df["format"].tolist()
        limitations = [f"{lmt}" for lmt in limitations]
    else:
        RuntimeError("lmt_type is invalid.")

    base_text_list = list()
    if "base_text" in master_df.columns:
        for base_text in master_df["base_text"].tolist():
            base_text_list.append(base_text)
    else:
        raise RuntimeError("invalid master dataframe")

    prompt_list = _get_prompt_list(base_text_list, limitations, task)
    return prompt_list

def get_header_footer_remover_prompt(task: str, generated_result: str) -> str:
    remove_prompt = ""
    if task == "summary":
        remove_prompt = f"""以下に提示している文章は、ある文章を生成AIを用いて要約した出力結果です。
出力には「要約」あるいはそれに類する単語を含むような文として、「以下の文章を要約します。」「【要約】」などの冒頭の説明文や「以上が要約結果になります。」などの文末の説明文が入っていることがあります。また、英語でこれらの説明文が与えられることもあります。
提示した文章に上記で述べた説明文が含まれていない場合には提示した文章をそのまま出力し、上記で述べた説明文が含まれている場合は提示した文章から説明文を除去したものを抜き出してください。文章の中間部分を編集する必要は一切ありません。文が入っていることがあります。また、英語でこれらの説明文が与えられることもあります。
[文章]
{generated_result}
"""
    elif task == "ad_text":
        remove_prompt = f"""以下に提示している文章は、ある文章を元に作成した広告文のタイトルです。
出力には「広告文：」や「広告文を作成します」などの冒頭の接頭辞や説明文、「作成しました。」「このタイトルは、、」などの接尾辞やタイトルの後ろの説明文が含まれていることがあります。
提示した文章に上記で述べた説明文や接頭辞、接尾辞が含まれていない場合には、提示した文章をそのまま出力してください。「」や**などの記号で囲われている事例の場合、記号は全て残したまま出力してください。
上記で述べた説明文が含まれている場合は提示した文章から説明文や接頭辞、接尾辞を除去したものを抜き出してください。冒頭、末尾以外の中間部分を編集する必要は一切ありません。新しく文字を追加などをしないでください。

[文章]
{generated_result}
"""
    elif task == "pros_and_cons":
        remove_prompt = f"""以下に提示している文章は、ある事象・事物についてのメリットとデメリットを生成AIに回答してもらった出力結果です。
文章の冒頭や末尾に「そこで、メリットとデメリットをご紹介いたします。」「あなたのご質問にお答えいたします。」「以上が〇〇に関するメリット・デメリットです。」など内容と関係のない説明文が付与されている場合は、その説明文を除去して出力してください。ただし、文の一部は変更せずに、該当の文全体を除去してください。
上記のような説明文が付与されていない場合は、提示している文章をそのまま出力してください。

[文章]
{generated_result}
"""
    return remove_prompt


def get_quality_check_prompt(task: str, generated_result: str, base_text: str = "") -> str:
    quality_check_prompt = ""
    if task == "summary":
        quality_check_prompt = f"""以下に要約した文章とその要約元の文章が提示されています。
要約した文章は要約元の文章を適切に要約できているかを判断してください。
適切に要約できている場合は「適切」、適切に要約できていない場合は「不適切」と回答してください。
ただし、要約元の文章から断定できない情報が要約した文章に含まれている場合も「不適切」と回答してください。
「適切」「不適切」のいずれかのみを出力し、説明文などは付与しないでください。
【要約元の文章】
{base_text}

【要約した文章】
{generated_result}
"""
    elif task == "ad_text":
        quality_check_prompt = f"""以下に、ランディングページの説明文とその説明文をもとに作成した1つの広告文のタイトルがあります。
説明文の内容に基づいているタイトルを作成できているかを判断してください。
適切に作成できている場合は「適切」、適切に作成できていない場合は「不適切」と回答してください。
ただし、説明文とタイトルが完全に一致している事例とタイトルとして長すぎる事例も「不適切」と回答してください。
「適切」「不適切」のいずれかのみを出力し、説明文などは付与しないでください。

【説明文】
{base_text}

【広告文のタイトル】
{generated_result}
"""
    elif task == "pros_and_cons":
        quality_check_prompt = f"""以下に提示している文章は、ある事象・事物についてのメリットとデメリットを生成AIに回答してもらった出力結果です。
出力結果が、メリット・デメリットの双方について言及できているか否かを回答してください。
言及できている場合は「適切」、言及できていない場合は「不適切」と回答してください。
「適切」「不適切」のいずれかのみを出力し、説明文などは付与しないでください。

【文章】
{generated_result}
"""
    return quality_check_prompt


def get_quality_check_prompt_bool(task: str, generated_result: str, base_text: str = "") -> str:
    quality_check_prompt = ""
    if task == "summary":
        quality_check_prompt = f"""以下に要約した文章とその要約元の文章が提示されています。
要約した文章は要約元の文章を適切に要約できているかを判断してください。
適切に要約できている場合はTrue、適切に要約できていない場合はFalseと回答してください。
ただし、要約元の文章から断定できない情報が要約した文章に含まれている場合もFalseと回答してください。
必ずTrue, Falseのいずれかの単語を1つだけ出力してください。
【要約元の文章】
{base_text}

【要約した文章】
{generated_result}
"""
    elif task == "ad_text":
        quality_check_prompt = f"""以下に、ランディングページの説明文とその説明文をもとに作成した1つの広告文のタイトルがあります。
説明文の内容に基づいているタイトルを作成できているかを判断してください。
適切に作成できている場合はTrue、適切に作成できていない場合はFalseと回答してください。
ただし、説明文とタイトルが完全に一致している事例とタイトルとして長すぎる事例もFalseと回答してください。
必ずTrue, Falseのいずれかの単語を1つだけ出力してください。

【説明文】
{base_text}

【広告文のタイトル】
{generated_result}
"""
    elif task == "pros_and_cons":
        quality_check_prompt = f"""以下に提示している文章は、ある事象・事物についてのメリットとデメリットを生成AIに回答してもらった出力結果です。
出力結果が、メリット・デメリットの双方について言及できているか否かを回答してください。
言及できている場合はTrue、言及できていない場合はFalseと回答してください。
必ずTrue, Falseのいずれかの単語を1つだけ出力してください。

【文章】
{generated_result}
"""
    return quality_check_prompt