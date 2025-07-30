from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import hf_hub_download
from langchain.llms.base import LLM
from typing import List, Optional
from langchain import PromptTemplate, LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import LlamaTokenizer
import torch
from langchain.prompts import PromptTemplate
import language_tool_python

def curate_with_languagetool(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text




with open(r"C:\Users\nteits\git_repos\DS_Test\audio_project\Transcriptions\transcriptions.txt", "r", encoding="utf-8") as f:
    input_text = f.read()  

result = curate_with_languagetool(input_text)



model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token='..'
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=1.2
)

llm = HuggingFacePipeline(pipeline=pipe)
correction_prompt = PromptTemplate.from_template(
    f"Correct the grammar and spelling of the following text.Only output the corrected version, nothing else:\n\n{input_text}\n"
)

correction_chain = LLMChain(llm=llm, prompt=correction_prompt)

corrected = correction_chain.invoke({"text": input_text})

