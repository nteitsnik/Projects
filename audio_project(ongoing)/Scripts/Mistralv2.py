from transformers import pipeline, AutoModelForCausalLM, LlamaTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def LLM_Use(model_name, max_new_tokens, temperature, input_text, instructions,tag, token=None):
    """
    Translates Greek text to English using a Hugging Face pipeline and LangChain.
    
    Args:
        model_name (str): The Hugging Face model name or path.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature for generation.
        input_text (str): The Greek text to translate.
        token (str, optional): Hugging Face authentication token if needed.
        
    Returns:
        str: The curated English translation, extracted after 'TRANSLATION:'.
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",  # or torch.float16 if you know your hardware supports it
        device_map="auto",
        token=token
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    # Use a prompt template with a variable for input text
    correction_prompt = PromptTemplate.from_template(
        "{instructions}\n\n{text}\n"
    )
    correction_chain = LLMChain(llm=llm, prompt=correction_prompt)

    # Use only the first 490 characters if needed
    
    corrected = correction_chain.invoke({
    "instructions": instructions,  # your instructions string
    "text": input_text             # your Greek text
     })

    # Extract the output string
    if isinstance(corrected, dict):
        corrected_text = corrected.get('text') or corrected.get('output') or ''
    else:
        corrected_text = corrected

    # Extract translation after 'TRANSLATION:'
    currated_output = corrected_text.split(f'{tag}')[-1].strip()
    return currated_output



def translate_en_el(model_name,input):
    '''Translates english to greek '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(input, return_tensors="pt")
    translated = model.generate(**inputs)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)

    return result







def full_pipeline(input_text):
    tag1='TRANSLATION :\n\n'

    instructions1 = (
        f"Translate the following Greek text in English.Be very careful on the semantics, it is really important.The translation should start with the word {tag1}:"
    )
 
    output1 = LLM_Use(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=1024,
    temperature=0.1,
    input_text=input_text,
    instructions=instructions1, tag=tag1,
    token='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'
    )
    
    tag2='OUTPUT :\n\n'
    instructions2=f"I want to summarize me the following text highlighting its key points.The output should start with {tag2}:"

    output2 = LLM_Use(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=1024,
    temperature=0.7,
    input_text=output1,
    instructions=instructions2, tag=tag2,
    token='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'
    )
    
    model_name = 'Helsinki-NLP/opus-mt-en-el'
    greek_translation = translate_en_el(model_name,output2)
    return greek_translation

