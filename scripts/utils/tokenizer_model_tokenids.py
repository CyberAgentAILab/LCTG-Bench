import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import google.generativeai as genai
from openai import OpenAI


def get_openai_model_tokenizer(model_path: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return client, model_path


def get_gemini_model_tokenizer(model_path: str):
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_path)
    return model, ""


def get_calm2_model_tokenizer(model_path:str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    return model, tokenizer


def get_elyza_model_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    return model, tokenizer


def get_rinna_gptneox_model_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # use_fast=True
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto") # device_map="auto", torch_dtype="auto" is not supported ???
    return model, tokenizer


def get_line_jallm_model_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path) # use_fast=False
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto") # device_map="auto", torch_dtype="auto" is not supported ???
    return model, tokenizer


def get_matsuo_model_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto") # device_map="auto", torch_dtype="auto" is not supported ???
    return model, tokenizer


def get_llmjp_model_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto") # device_map="auto", torch_dtype="auto" is not supported ???
    return model, tokenizer


def get_rinna_youri_model_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_type="auto") # device_map="auto", torch_dtype="auto" is not supported ???
    return model, tokenizer


def get_stabilityai_stablelm_instruct_gamma_model_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    return model, tokenizer


def get_swallow_model_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    return model, tokenizer

