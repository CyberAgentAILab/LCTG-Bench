# LCTG Bench: LLM Controlled Text Generation Benchmark
The LCTG Bench has been built to measure the controllability of Japanese LLMs in terms of how well they comply with constraints such as character count keywords in instructions.


## Task/Datasets
The LCTG Bench consists of three text generation tasks: **Summarization**, **Advertisement (AD) Text Geneartion**, and **Pros & Cons Generation**. For each task, the performance of LLM is evaluated in terms of four controllability aspects: `Format`, `Character Count`, `Keyword`, and `Prohibited word`.

All data sets consist of test data only.

The Pros & Cons Generation task has been built after writing the Japanese paper.

LCTG Bench has been constructed in collaboration with AI Shift Inc., CyberAgent Inc. and Okazaki Lab at Tokyo Institute of Technology.

| Task                   | Dataset     | Format | Character Count | Keyword | Prohibihited word |
|------------------------|-------------|--------|-----------------|---------|-------------------|
| Summarization          | ABEMA Times | 120    | 120             | 120     | 120               |
| Ad Text Generation     | CAMERA      | 150    | 150             | 150     | 150               |
| Pros & Cons Generation | -           | 150    | 150             | 150     | 150               |

## Evaluation Method
You can run the LLM performance evaluation using this benchmark simply by executing ```run_lctg.sh```.
All scripts are placed under ```scripts```, and datasets are placed under ```datasets```.

### Procedure
0. To run this script, you will need to prepare a python 3.10 operating environment in advance. (We have not verified that it works with python 3.11 or later versions.)
1. To install packages, run the following command.
```angular2html
pip install -r requirements.txt
```

2. To conduct an evaluation, you must prepare a ```.env``` file with ```OPENAI_API_KEY```, ```VERSION```, and ```MODEL_PATH_X``` (where X= 0, 1, 2 ...) defined (please specify "1" for ```VERSION```). For ```MODEL_PATH_X```, specify the path of the model on the hugging face hub or the name of the model provided by the API. ***Please always specify X as a sequential number starting from 0.***
```angular2html
OPENAI_API_KEY={your_openai_api_key}
VERSION="1"
MODEL_PATH_0="gpt-4-1106-preview"
MODEL_PATH_1="cyberagent/calm2-7b-chat"
```
3. By executing the following command, inference by LLM and evaluation of the inference results will be conducted. After execution, you can check the scores at ```output/{task}/all_scores.csv```. It takes several hours to execute the evaluation for each model.
```
make run
```

### Notes
Currently, this repository covers the following models.
```
MODEL_PATH_0="gpt-4-1106-preview"
MODEL_PATH_1="gpt-3.5-turbo-0125"
MODEL_PATH_2="gemini-pro"
MODEL_PATH_3="cyberagent/calm2-7b-chat"
MODEL_PATH_4="elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"
MODEL_PATH_5="line-corporation/japanese-large-lm-3.6b-instruction-sft"
MODEL_PATH_6="matsuo-lab/weblab-10b-instruction-sft"
MODEL_PATH_7="rinna/youri-7b-chat"
MODEL_PATH_8="stabilityai/japanese-stablelm-instruct-gamma-7b"
MODEL_PATH_9="llm-jp/llm-jp-13b-instruct-full-jaster-v1.0"
MODEL_PATH_10="llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0"
MODEL_PATH_11="tokyotech-llm/Swallow-7b-instruct-hf"
MODEL_PATH_12="tokyotech-llm/Swallow-13b-instruct-hf"
MODEL_PATH_13="tokyotech-llm/Swallow-70b-instruct-hf"
```
If you wish to evaluate the performance of models other than those listed above, you will need to edit the following files in addition to adding environment variables.
But if you want to use the openai model. You only have to edit ```scripts/utils.model_info_getter.py```

#### scripts/utils/tokenizer_model_tokenids.py
- you have to add a function that returns model and tokenizer
```
def get_calm2_model_tokenizer(model_path:str):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
```

#### scripts/utils/output_tokens.py
- You have to add a function that returns the output tokens (str).
- You may also have to define system prompts and model parameters in the functions you add.
```
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
```

#### scripts/utils.model_info_getter.py
- You have to edit the following two functions.
  - get_model_tokenizer
    - you have to add few lines.
```
    elif "cyberagent/calm2" in model_path:
        model, tokenizer = get_calm2_model_tokenizer(model_path)
```
  - get_output_tokens
    - You have to add the pair of key (model name) and value (function name)
```
"cyberagent/calm2-7b-chat": get_calm2_output_tokens,
```

## Citation 
If you use it in your research, please cite:

```
@InProceedings{Kurihara_nlp2024,
  author = 	"栗原健太郎 and 三田雅人 and 張培楠 and 佐々木翔大 and 石上亮介 and 岡崎直観",
  title = 	"LCTG Bench: 日本語LLMの制御性ベンチマークの構築",
  booktitle = 	"言語処理学会第30回年次大会",
  year =	"2024",
  url = "https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/D11-2.pdf"
  note= "in Japanese"
}
```
## License
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0//"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />

## Prohibited Uses
The evaluation results from this benchmark may be made available to the public, but the following uses are prohibited

- Mentioning Abema TV Inc. as the provider of ABEMA TIMES, the data source for the summarization task
