from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

from langchain.llms.base import LLM
from typing import List, Optional

class TinyLlamaLLM(LLM):
    def __init__(self, generator, stop: Optional[List[str]] = None):
        self.generator = generator
        self.stop = stop

    @property
    def _llm_type(self) -> str:
        return "tinyllama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        output = self.generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]['generated_text']
        return output[len(prompt):]  # Remove prompt from output
    
from langchain.agents import initialize_agent, load_tools
from langchain.agents.agent_types import AgentType   
from pydantic import Field
from typing import Any


class TinyLlamaLLM(LLM):
    generator: Any = Field(exclude=True)
    stop: Optional[List[str]] = None

    @property
    def _llm_type(self) -> str:
        return "tinyllama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Format the instruction prompt
        instruction = f"""### Instruction:
Correct the following sentence syntactically, grammatically, and semantically:
{prompt}

### Response:"""

        output = self.generator(
            instruction,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.9
        )[0]['generated_text']

        return output[len(instruction):].strip()

with open(r"C:\Users\Nteit\audio_project\Transcriptions\transcriptions.txt", "r", encoding="utf-8") as f:
    content = f.read()  

llm = TinyLlamaLLM(generator=generator)

corrected = llm(content)
