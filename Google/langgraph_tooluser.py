"""
Source: https://github.com/langchain-ai/langgraph/blob/main/examples/chat_agent_executor_with_function_calling/dynamically-returning-directly.ipynb
WHY?: Graphs give us more control over the intermediate steps
The AgentExecutor alone does not function realiably
Graphs allow us to seperate tool use and analysis
Graphs can be drawn and analyzed better
"""
#############################################################################
# Imports
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
##########################################################################
# SET UP TOKENS
from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]
TAVILY_API_KEY = config_data["TAVILY_API_KEY"]
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = config_data["LANGCHAIN_API_KEY"]
os.environ["GOOGLE_API_KEY"] = config_data["GOOGLE_API_KEY"]
##########################################################################
# SET UP TOOLS 
from langchain_core.pydantic_v1 import BaseModel, Field
# args schema
class SearchTool(BaseModel):
    """Look up things online, optionally returning directly"""

    query: str = Field(description="query to look up online")
    return_direct: bool = Field(
        description="Whether or the result of this should be returned directly to the user without you seeing what it is",
        default=False,
    )
from langchain_community.tools.tavily_search import TavilySearchResults
# search tool
search_tool = TavilySearchResults(max_results=1, args_schema=SearchTool)
tools = [search_tool]
from langgraph.prebuilt import ToolExecutor
# tool executor (calls tool -> returns output)
tool_executor = ToolExecutor(tools)

##########################################################################
# SET UP MODEL
from langchain_google_vertexai import ChatVertexAI
llm = ChatVertexAI(model="gemini-pro")

##########################################################################
# DEFINE AGENT STATE
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

###########################################################################
# DEFINE NODES
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import ToolMessage
# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we check if it's suppose to return direct
    else:
        arguments = last_message.tool_calls[0]["args"]
        if arguments.get("return_direct", False):
            return "final"
        else:
            return "continue"
# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}
# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    arguments = tool_call["args"]
    if tool_name == "tavily_search_results_json":
        if "return_direct" in arguments:
            del arguments["return_direct"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=arguments,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a ToolMessage
    tool_message = ToolMessage(
        content=str(response), name=action.tool, tool_call_id=tool_call["id"]
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [tool_message]}

###########################################################################
# DEFINE GRAPH
from langgraph.graph import StateGraph, END

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.add_node("final", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Final call
        "final": "final",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")
workflow.add_edge("final", END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
for output in app.stream(inputs):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")