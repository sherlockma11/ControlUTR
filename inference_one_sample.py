import csv
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import torch
from ViennaRNA import RNA
import LLaMA_Factory.src.llamafactory.train.ppo4mrna.RNABenchmark
from RNABenchmark.model.splicebert.modeling_splicebert import SpliceBertForSequenceClassification
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer
import transformers


### For demonstration purposes, only one sample is generated.


model_name = "Qwen2.5-0.5b-mrna-stage3-dpo-1-ch1000"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'


# prompt1 = "你是一位在生物信息学和分子生物学领域有着深厚背景的专家，擅长解析生物属性与基因表达之间的关系，并能够根据这些属性设计相应的mRNA序列。\n以下是一些案例：mRNA的MRL=-1.7812239923226838，对应的mRNA序列是ACCAACATGTAATTTCCACTCTTGA\nmRNA的MRL=-0.275642799200671，对应的mRNA序列是TGGTAAAATCTAGGGTTTTTTATAA\nmRNA的MRL=-1.1482258696972798，对应的mRNA序列是CAAAAAGTAGACGCAACATGAAAAA\n\n现在我们想要你根据性质生成对应的mRNA，它的MRL="
prompt2 = "，请生成对应的mRNA序列."
prompt1_len1 = "你是一位在生物信息学和分子生物学领域有着深厚背景的专家，擅长解析生物属性与基因表达之间的关系，并能够根据这些属性设计相应的mRNA序列。不要生成‘A’，‘T’，‘C’，‘G’之外的字符\n现在我们想要你根据性质生成对应的mRNA，它的MRL="
prompt1_len2 = "，长度是"
prompt1_len3 = '，MFE是'

mrls = []
mfes = []
lengths = []
texts = []
mrl = round(random.uniform(1.2, 1.6), 2)
mfe = round(random.uniform(-25, -15), 2)
length = str(random.randint(20,100))
mrls.append(mrl)
mfes.append(mfe)
lengths.append(length)
messages = [
    # {"role": "user", "content": prompt1+mrl+prompt2}      # mrl
    {"role": "user", "content": prompt1_len1 + str(mrl) + prompt1_len2 + length + prompt1_len3 + str(mfe) + prompt2}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
texts.append(text)

model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    # temperature=1.2
)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
response = [response[i].split("assistant\n")[-1] for i in range(len(response))]

## mfe
mfe_values = []
for mrna in response:
    # 创建一个折叠复合体
    fc = RNA.fold_compound(mrna)

    # 计算MFE和相应的二级结构
    structure, mfe_value = fc.mfe()
    mfe_values.append(mfe_value)


## mrl
tokenizer_mrl = OpenRnaLMTokenizer.from_pretrained(
    "<You Path>",
    cache_dir=None,
    model_max_length=512,
    padding_side="right",
    use_fast=True,
    trust_remote_code=True,
)
model_mrl = SpliceBertForSequenceClassification.from_pretrained(
    "<You Path>",
    cache_dir=None,
    num_labels=1,
    problem_type="regression",
    trust_remote_code=True,
)
model_mrl = model_mrl.half()
model_mrl.to("cuda")
with torch.no_grad():
    t = tokenizer_mrl(response, padding=True, truncation=True, return_tensors="pt").to("cuda")
    predictions = model_mrl.forward(**t)
    logits = [i[0].item() for i in predictions.logits]



print("Generated UTR sequence:\t", response[0],
      "\nThe expected MRL value is:\t", mrls[0], "The obtained MRL value is：\t", logits[0], "The error is：\t", abs(logits[0]-mrls[0]),
      "\nThe expected MFE value is:\t", mfes[0], "The obtained MFE value is：\t", mfe_values[0], "The error is：\t", abs(mfe_values[0]-mfes[0]))


### Output Reference
"""
Generated UTR sequence:	 GTTCGCATTGCCAAACAAGCAGTGCGAACGATTAACGAGCACAGTTTTGCGAGAGAAAG 
The expected MRL value is:	 1.56 The obtained MRL value is：	 1.4970703125 The error is：	 0.06292968750000005 
The expected MFE value is:	 -20.57 The obtained MFE value is：	 -19.700000762939453 The error is：	 0.8699992370605472
"""
