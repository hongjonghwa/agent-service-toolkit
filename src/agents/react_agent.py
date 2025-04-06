from pprint import pprint
from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

from agents.tools import calculator  # 예시 도구
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """Represents the state of our ReAct agent.

    Attributes:
        messages: The history of messages in the conversation.
    """
    # 필요한 경우 여기에 추가 상태 속성을 정의할 수 있습니다.
    # 예: intermediate_steps: list = []
    is_last_step: IsLastStep


# 사용할 도구를 정의합니다. 필요에 따라 추가하거나 수정하세요.
tools = [calculator, DuckDuckGoSearchResults(name="WebSearch")] # 예시로 계산기와 웹 검색 도구 추가

# 에이전트의 시스템 프롬프트/지침
# ReAct 스타일에 맞게 생각 과정을 명시하도록 유도할 수 있습니다.
instructions = f"""
You are a helpful assistant that thinks step-by-step to solve problems.
Today's date is {datetime.now().strftime("%B %d, %Y")}.

When you need to use a tool, explain your reasoning (Thought), specify the tool to use and the input (Action), and then wait for the result (Observation).
If you have enough information to answer, provide the final answer directly.

Example format:
Thought: I need to calculate 5 * 5.
Action: calculator(expression="5*5")

Thought: I need to search the web for the capital of France.
Action: WebSearch(query="capital of France")

Respond with the final answer once you have it.
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wraps the language model with system instructions and tool binding."""
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Invokes the language model and returns the updated state."""
    # 설정에서 모델 가져오기
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    # 모델 호출
    response = await model_runnable.ainvoke(state, config)


    # Handle the case when it's the last step and the model still wants to use a tool
    # pprint(state)
    # print("state.is_last_step", state.get('is_last_step'), "response.tool_calls", response.tool_calls)
    if state.get('is_last_step') and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # 상태 업데이트 (새 메시지 추가)
    return {"messages": [response]}


# 도구 실행 노드
tool_node = ToolNode(tools)


# 모델 호출 후 도구를 실행할지 또는 종료할지 결정하는 함수
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determines whether to continue with tool execution or end the process."""
    last_message = state["messages"][-1]
    # 마지막 메시지가 AI 메시지이고 도구 호출이 있으면 'tools' 반환
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    # 그렇지 않으면 종료
    return "__end__"


# 그래프 정의
agent_graph = StateGraph(AgentState)

# 노드 추가: 모델 호출 노드와 도구 실행 노드
agent_graph.add_node("model", acall_model)
agent_graph.add_node("tools", tool_node)

# 진입점 설정
agent_graph.set_entry_point("model")

# 조건부 엣지 추가: 모델 호출 후 상태에 따라 분기
agent_graph.add_conditional_edges(
    "model",
    should_continue,
    {
        "tools": "tools",  # 도구 호출이 있으면 'tools' 노드로 이동
        "__end__": END,    # 도구 호출이 없으면 종료
    },
)

# 엣지 추가: 도구 실행 후에는 항상 'model' 노드로 돌아감
agent_graph.add_edge("tools", "model")

# 그래프 컴파일
react_agent = agent_graph.compile()
react_agent.name = "ReAct 에이전트"  # This customizes the name in LangSmith

# # 에이전트 실행 예시 (주석 처리됨)
# async def run_agent(query: str):
#     async for event in react_agent.astream_events(
#         {"messages": [("user", query)]},
#         version="v1"
#     ):
#         kind = event["event"]
#         if kind == "on_chat_model_stream":
#             content = event["data"]["chunk"].content
#             if content:
#                 print(content, end="|")
#         elif kind == "on_tool_start":
#             print("--")
#             print(f"Starting tool: {event['name']} with input: {event['data'].get('input')}")
#         elif kind == "on_tool_end":
#             print(f"Tool ended: {event['name']}")
#             print(f"Tool output: {event['data'].get('output')}")
#             print("--")
#         elif kind in ["on_chat_model_start", "on_chain_start", "on_chain_end", "on_tool_start", "on_tool_end"]:
#             pass # 필요시 로깅 추가
#         else:
#             print(event) # 기타 이벤트 로깅

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(run_agent("What is 5 * 5?"))
#     asyncio.run(run_agent("What is the weather in Seoul today?")) # Requires OpenWeatherMap API key configured
