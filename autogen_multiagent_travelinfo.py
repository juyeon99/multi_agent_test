# Ref: https://youtu.be/qwOOOLX5vSo?feature=shared
# autogenstudio ui --port 8080 --appdir ./my_app
import asyncio, logging
from typing import Callable, List, Union

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# tools 정의
def search_web_tool(query: str) -> str:
    logger.info(f"웹 검색 도구 호출: {query}")
    if "서울" in query and "2021" in query:
        return """2021년 서울의 주요 관광지 방문자 수:
        경복궁: 2,500,000명
        남산타워: 1,800,000명
        홍대: 3,200,000명
        """
    elif "2022" in query:
        return "2022년 서울의 주요 관광지 방문자 수 증가율에 대한 데이터는 아직 준비 중 입니다."
    return "데이터를 찾을 수 없습니다."

def percentage_change_tool(start: float, end: float) -> float:
    logger.info(f"퍼센트 변화 계산 도구 호출: 시작={start}, 끝={end}")
    return ((end - start) / start) * 100

# OpenAI 챗 컴플리션 클라이언트 초기화
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# 계획 수립 에이전트 정의
planning_agent = AssistantAgent(
    "PlanningAgent",
    description="작업을 계획하는 에이전트로, 새로운 작업이 주어졌을 때 가장 먼저 참여해야 합니다.",
    model_client=model_client,
    system_message="""
        당신은 계획 수립 에이전트 입니다.
        복잡한 작업을 더 작고 관리 가능한 하위 작업으로 분해하는 것이 당신의 업무입니다.
        당신의 팀원은 다음과 같습니다:
            웹 검색 에이전트: 정보를 검색합니다.
            데이터 분석가: 계산을 수행합니다.
        
        당신은 작업을 계획하고 위임만 할 수 있으며, 스스로 실행하지 않습니다.

        작업을 할당할 때는 다음 형식을 사용하세요.
        1. <에이전트> : <작업>

        모든 작업이 완료된 후, 결과를 요약하고 "TERMINATE"로 끝냅니다.
        """
)

# 웹 검색 에이전트 정의
web_search_agent = AssistantAgent(
    "WebSearchAgent",
    description="웹 검색을 담당하는 에이전트 입니다.",
    tools=[search_web_tool],
    model_client=model_client,
    system_message="""
        당신은 웹 검색 에이전트입니다.
        당신의 유일한 도구는 search_web_tool 이며, 이를 사용하여 정보를 찾으세요.
        한 번에 하나의 검색만 수행할 수 있습니다.
        결과를 받은 후에는 이를 기반으로 계산을 하지 않습니다.
        """
)

# 데이터 분석가 에이전트 정의
data_analyst_agent = AssistantAgent(
    "DataAnalystAgent",
    description="계산을 수행하는 데이터 분석가 에이전트 입니다.",
    tools=[percentage_change_tool],
    model_client=model_client,
    system_message="""
        당신은 데이터 분석가입니다.
        할당한 작업을 바탕으로 데이터를 분석하고 제공된 도구를 사용하여 결과를 도출해야 합니다.
        """
)

# 종료 조건 정의
text_mention_termination = TextMentionTermination("TERMINATE")
max_message_termination = MaxMessageTermination(max_messages=20)
termination = text_mention_termination | max_message_termination

# 팀 구성 및 종료 조건 설정
team = SelectorGroupChat(
    [planning_agent, web_search_agent, data_analyst_agent],
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
    termination_condition=termination
)

# 여행 관련 작업 정의
# task = "서울에서 가장 많이 방문한 관광지는 어디이며, 2021년과 2022년 사이에 방문자 수가 몇 퍼센트 증가했는지 알려주세요."
task = "2022년에 서울에서 가장 많이 방문한 관광지를 순서대로 알려주세요."

async def main():
    try:
        await Console(team.run_stream(task=task))
    except Exception as e:
        logger.error(f"오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main())