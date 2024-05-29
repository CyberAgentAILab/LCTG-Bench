import logging

import torch

MAX_NEW_TOKENS = 4096


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def get_openai_output_tokens(model, tokenizer, prompt, use_system_prompt: bool = True):
    model_path = tokenizer
    if use_system_prompt:
        system_prompt = "あなたは指示に忠実に従うAIアシスタントです。ユーザーの指示に従って下さい。"
    else:
        system_prompt = ""
    completion = model.chat.completions.create(
        model=model_path,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    output_tokens = completion.choices[0].message.content
    return output_tokens


def get_gemini_output_tokens(model, tokenizer, prompt, use_system_prompt: bool = True):
    response = model.generate_content(
                    prompt,
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE"
                        },
                    ]
                )
    try:
        output_tokens = response.text
    except ValueError:
        output_tokens = f"ValueError: {response.prompt_feedback}"
        logger.info(output_tokens)
    except IndexError:
        output_tokens = "IndexError"
        logger.info(output_tokens)
    return output_tokens


def get_calm2_output_tokens(model, tokenizer, text: str, use_system_prompt: bool = True):
    prompt = f"""USER: {text}
ASSISTANT: """
    token_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=token_ids.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
        )
        output_ids = output_ids[0][len(token_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def get_elyza_output_tokens(model, tokenizer, text: str, use_system_prompt: bool = True):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    if use_system_prompt:
        DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
    else:
        DEFAULT_SYSTEM_PROMPT = ""
    prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=text,
        e_inst=E_INST,
    )
    with torch.no_grad():
        token_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        output_ids = output_ids[0][len(token_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def get_rinna_gptneox_output_tokens(model, tokenizer, text: str, use_system_prompt: bool = True):
    if use_system_prompt:

        prompt = [
            {
                "speaker": "ユーザー",
                "text": "Hello, you are an assistant that helps me learn Japanese."
            },
            {
                "speaker": "システム",
                "text": "Sure, what can I do for you?"
            },
        ]
    else:
        prompt = [
            {
                "speaker": "ユーザー",
                "text": f"{text}"
            }
        ]

    prompt = [f"{uttr['speaker']}: {uttr['text']}" for uttr in prompt]
    prompt = "\n".join(prompt)
    prompt = (prompt + "\n" + "システム: ")

    """
    ユーザー: Hello, you are an assistant that helps me learn Japanese.
    システム: Sure, what can I do for you?
    ユーザー: {text}
    システム:
    """
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_p=0.85,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        output_ids = output_ids[0][len(token_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def get_line_jallm_output_tokens(model, tokenizer, text: str, use_system_prompt: bool = True):
    prompt = f"ユーザー: {text}\nシステム: "
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.1,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
        output_ids = output_ids[0][len(token_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def get_matsuo_output_tokens(model, tokenizer, text: str, use_system_prompt: bool = True):
    if use_system_prompt:
        text = f"以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{text}\n\n### 応答:"
    else:
        text = f"指示:\n{text}\n\n### 応答:"
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        output_ids = output_ids[0][len(token_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def get_rinna_youri_output_tokens(model, tokenizer, text: str, use_sys: bool = True):
    if use_sys:
        instruction = "あなたは指示に忠実に従うAIアシスタントです。ユーザーの指示に従って下さい。"
    else:
        instruction = ""
    context = [
        {
            "speaker": "設定",
            "text": instruction
        },
        {
            "speaker": "ユーザー",
            "text": text
        }
    ]
    prompt = [f"{uttr['speaker']}: {uttr['text']}" for uttr in context]
    prompt = "\n".join(prompt)
    prompt = (prompt + "\n" + "システム: ")
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.5,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        output_ids = output_ids[0][len(token_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def get_llmjp_output_tokens(model, tokenizer, text: str, use_system_prompt: bool = True):
    text = text + "### 回答："
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            token_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
        )
        output_ids = output_ids[0][len(token_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def get_stabilityai_stablelm_instruct_gamma(model, tokenizer, text: str, use_system_prompt: bool = True):
    if use_system_prompt:
        sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    else:
        sys_msg = ""
    sep = "\n\n### "
    p = sys_msg
    roles = ["指示", "応答"]
    msgs = [": \n" + text, ": \n"]

    for role, msg in zip(roles, msgs):
        p += sep + role + msg

    input_ids = tokenizer.encode(p, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.to(device=model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1,
            top_p=0.95,
            do_sample=True
        )
        output_ids = output_ids[0][len(input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def get_swallow_output_tokens(model, tokenizer, text: str, use_system_prompt: bool = True):
    if use_system_prompt:
        prompt = f"""以下に、あるタスクを説明する指示があります。リクエストを適切に完了するための回答を記述してください。
    
### 指示:
{text}
    
### 応答:"""
    else:
        prompt = text

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.to(device=model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.99,
            top_p=0.95,
            do_sample=True,
        )
        output_ids = output_ids[0][len(input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)
