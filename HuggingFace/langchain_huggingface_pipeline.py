from langchain_huggingface import HuggingFacePipeline
import json
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          pipeline)
import accelerate
from langchain_huggingface import HuggingFacePipeline
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain import hub
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.tools.render import render_text_description
from langchain_community.utilities import SerpAPIWrapper
import gradio as gr
###########################################################################
# SET UP TOKENS
config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]

##########################################################################
# Create llm
model_name = "meta-llama/Meta-Llama-3-8B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"":0}, #requires and uses the Accelerate library
    quantization_config=bnb_config,
    token=HF_TOKEN,
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128
)
llm_4bit = HuggingFacePipeline(pipeline=pipe)
llm_4bit.invoke("Hugging Face is")
#######################################################################
# Create Chain