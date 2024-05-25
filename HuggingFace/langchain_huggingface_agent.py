"""
Agent with tool use.
HuggingFaceEndpoint. Langchain ChatModel.
works with any model inside a dedicated HuggingFace Endpoint.
Llama3-8B ~ 1$ / hour
"""

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
import os
###########################################################################
# SET UP TOKENS
config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]
TAVILY_API_KEY = config_data["TAVILY_API_KEY"]
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
#set up langsmith 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = config_data["LANGCHAIN_API_KEY"]
##########################################################################
# setup tools
from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool

tavily_tool = TavilySearchResults(max_results=5)

# Warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()
@tool
def python_repl(
    code: Annotated[str, "The python code to execute."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )
python_repl = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
    return_direct=True,
)
tools = [tavily_tool, python_repl]

##############################################################################
# Create llm with inference endpoint
from langchain_community.llms import HuggingFaceEndpoint

ENDPOINT_URL = "https://kpz14qiv0qxknusm.us-east-1.aws.endpoints.huggingface.cloud"
llm = HuggingFaceEndpoint(
    endpoint_url=ENDPOINT_URL,
    task="text-generation",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HF_TOKEN
)

#llm.invoke("Hugging Face is")
###########################################################################
# Create ChatModel
chat_model = ChatHuggingFace(llm=llm)
###########################################################################
# Define tools, agent, agent executor 
# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# define the agent
chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

# instantiate AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke(
    {
        "input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
    }
)
###########################################################################
# Create agent