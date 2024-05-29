from .tokenizer_model_tokenids import *
from .output_tokens import *


def get_suffix(model_path: str):
    return model_path.replace("/", "-")


def get_model_tokenizer(model_path: str):
    if "cyberagent/calm2" in model_path:
        model, tokenizer = get_calm2_model_tokenizer(model_path)
    elif "elyza/ELYZA-japanese-" in model_path:
        model, tokenizer = get_elyza_model_tokenizer(model_path)
    elif "rinna/bilingual-gpt-" in model_path:
        model, tokenizer = get_rinna_gptneox_model_tokenizer(model_path)
    elif "line-corporation/japanese-large-lm-" in model_path:
        model, tokenizer = get_line_jallm_model_tokenizer(model_path)
    elif "matsuo-lab/weblab-10b-instruction-sft" in model_path:
        model, tokenizer = get_matsuo_model_tokenizer(model_path)
    elif "llm-jp/llm-jp-13b-instruct-full-" in model_path:
        model, tokenizer = get_llmjp_model_tokenizer(model_path)
    elif "rinna/youri-7b-chat" in model_path:
        model, tokenizer = get_rinna_youri_model_tokenizer(model_path)
    elif "stabilityai/japanese-stablelm-instruct-gamma-7b" in model_path:
        model, tokenizer = get_stabilityai_stablelm_instruct_gamma_model_tokenizer(model_path)
    elif "tokyotech-llm/Swallow-" in model_path:
        model, tokenizer = get_swallow_model_tokenizer(model_path)
    elif model_path in ["gemini-pro"]:
        model, tokenizer = get_gemini_model_tokenizer(model_path)
    elif model_path in ["gpt-4-1106-preview", "gpt-3.5-turbo-0125"]:
        model, tokenizer = get_openai_model_tokenizer(model_path)
    else:
        raise RuntimeError("想定していないモデルを指定しています。")
    return model, tokenizer


def get_output_tokens(model_path: str, model, tokenizer, prompt: str, use_system_prompt: bool = True):
    model_to_output_tokens_getter = {
        "cyberagent/calm2-7b-chat": get_calm2_output_tokens,
        "rinna/bilingual-gpt-neox-4b-instruction-ppo": get_rinna_gptneox_output_tokens,
        "rinna/bilingual-gpt-neox-4b-instruction-sft": get_rinna_gptneox_output_tokens,
        "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct": get_elyza_output_tokens,
        "line-corporation/japanese-large-lm-3.6b-instruction-sft": get_line_jallm_output_tokens,
        "matsuo-lab/weblab-10b-instruction-sft": get_matsuo_output_tokens,
        "llm-jp/llm-jp-13b-instruct-full-jaster-v1.0": get_llmjp_output_tokens,
        "llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0": get_llmjp_output_tokens,
        "llm-jp/llm-jp-13b-instruct-full-dolly-oasst-v1.0": get_llmjp_output_tokens,
        "rinna/youri-7b-chat": get_rinna_youri_output_tokens,
        "stabilityai/japanese-stablelm-instruct-gamma-7b": get_stabilityai_stablelm_instruct_gamma,
        "tokyotech-llm/Swallow-7b-instruct-hf": get_swallow_output_tokens,
        "tokyotech-llm/Swallow-13b-instruct-hf": get_swallow_output_tokens,
        "tokyotech-llm/Swallow-70b-instruct-hf": get_swallow_output_tokens,
        "gemini-pro": get_gemini_output_tokens,
        "gpt-4-1106-preview": get_openai_output_tokens,
        "gpt-3.5-turbo-0125": get_openai_output_tokens
    }
    if model_path not in model_to_output_tokens_getter.keys():
        raise RuntimeError("A model that does not exist in the candidates is specified.")
    output_tokens = model_to_output_tokens_getter[model_path](model, tokenizer, prompt, use_system_prompt)
    return output_tokens