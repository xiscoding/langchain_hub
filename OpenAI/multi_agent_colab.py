"""
YOUTUBE: https://www.youtube.com/watch?v=hvAPnpSfSGo
@author: LANGCHAIN
@author: xdoestech (added human input)

cost: roughly 10-30cents per run (gpt-4o, gpt-4 turbo)
"""
############################################################################
# SET UP ENVIRONMENT
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY")

##############################################################################
# Helper functions
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph


def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " ask the user for confirmation (use the human_tool)."
                " If the user responds FINISH. Prefix your response with FINAL ANSWER."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)
#########################################################################
# Tool Def
from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(max_results=5)

# Warning: This executes code locally, which can be unsafe when not sandboxed

repl = PythonREPL()
@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."]
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

from langchain.pydantic_v1 import BaseModel, Field
# NOTE: args_schema forces model to use correct tool input
class HumanInput(BaseModel):
    query: str = Field(description="should be a query")

from langchain_community.tools import HumanInputRun
# humanAsATool
human_tool_desc = '''
You can use this tool to ask the user for the details related to the request.
Always use this tool if you have follow-up questions.
Be concise, polite and professional when asking the questions.
Args:
  query: Question to ask user
'''

def get_input():
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)
human_tool = HumanInputRun(input_func=get_input, description=human_tool_desc, args_schema=HumanInput)
################################################################################
# Create Graph
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

## STATE
# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
## AGENT NODES
import functools
from langchain_core.messages import AIMessage


# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    #If tool continues, else send message to AI
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

gpt3_5 = "gpt-3.5-turbo-0125"
gpt4_o = "gpt-4o"
llm = ChatOpenAI(model=gpt4_o)

#create_agent defined above
# Research agent and node
research_agent = create_agent(
    llm,
    [tavily_tool, human_tool],
    system_message="You should provide accurate data for the chart_generator to use.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# chart_generator
chart_agent = create_agent(
    llm,
    [python_repl, human_tool],
    system_message="Any charts you display will be visible by the user.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

## TOOL NODE
from langgraph.prebuilt import ToolNode

tools = [tavily_tool, python_repl, human_tool]
tool_node = ToolNode(tools)

## EDGE LOGIC
# Either agent can decide:
# Invoke tool
# Final answer
# continue to other AI Agent
from typing import Literal

#ADD TOOLS HERE
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        #TODO: add human approval
        return "__end__"
    return "continue"

## GRAPH 
#1. Create Graph with AgentState (messages and sender information)
workflow = StateGraph(AgentState)

#2. Add Agent Nodes
workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

#3. Reasearcher/Chart Generator user router logic, call tool, continue, end
workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "chart_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
)

workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "chart_generator": "chart_generator",
    },
)
workflow.set_entry_point("Researcher")
graph = workflow.compile()

## VIEW GRAPH
# Image(graph.get_graph(xray=True).draw_mermaid_png())
##############################################################################
# INVOKE 
events = graph.stream( #stream shows output
    {
        "messages": [
            HumanMessage(
                content="Fetch the UK's GDP over the past 5 years,"
                " then draw a line graph of it."
                " Once you code it up, use the human_tool to ask me if you can finish."
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 20},
)
for s in events:
    print(s)
    print("----")