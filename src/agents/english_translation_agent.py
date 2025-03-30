from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator
from core import get_model, settings

from langchain.chat_models import init_chat_model
from datetime import datetime, timezone


class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    """ 굳이 필요 없다. 
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    messages = []

    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """

class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    # Additional attributes can be added here as needed.
    # Common examples include:
    # is_last_step: IsLastStep = field(default=False)
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)





class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


tools = []

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

instructions = """
You are an AI translator to english.
Don't translate the sentence literally, 
Summarize the meaning in short and simple sentences.
"""



def get_prompt_messages(messages: list):
    # other_msgs = []
    # system_prompt = SYSTEM_PROMPT
    # for m in messages:
    #     if isinstance(m, SystemMessage):
    #         system_prompt = m.content
    #     else:
    #         other_msgs.append(m)
    return [SystemMessage(content="(system prompt)")] + messages


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


    # fully_specified_name = "anthropic/claude-3-7-sonnet-20250219"
    # provider, model = fully_specified_name.split("/", maxsplit=1)
    # model_runnable = init_chat_model(model, model_provider=provider).bind_tools([])

    # # response = await model_runnable.ainvoke(state, config)
    # response = await model_runnable.ainvoke(get_prompt_messages(state["messages"]), config)

    # # We return a list, because this will get added to the existing list
    # return {"messages": [response]}

# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.set_entry_point("model")
agent.add_edge("model", END)

english_translation_agent = agent.compile(checkpointer=MemorySaver())
