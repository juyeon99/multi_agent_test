import asyncio
from typing import Callable, List, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

# 도구 정의
async def get_travel_info(destination:str) -> str:
    # 실제 구현에선 API 호출 등을 통해 실시간 정보를 가져올 수 있음
    # 여기선 예시로 하드코드된 응답 반환
    travel_data= {
        "서울" : "서울의 주요 관광지로는 경복궁, 남산타워, 홍대 등이 있으며, 2025년 현재 다양한 축제와 문화 행사가 열리고 있습니다.",
        "부산" : "부산의 주요 관광지로는 해운대 해수욕장, 광안리 해수욕장, 자갈치 시장 등이 있습니다. 매년 많은 관광객이 방문합니다.",
        "제주도" : "제주도의 주요 관광지로는 한라산, 성산일출봉, 제주 올레길 등이 있으며, 아름다운 자연 경관으로 유명합니다."
    }

    return travel_data.get(destination, "해당 지역에 대한 여행 정보를 찾을 수 업습니다.")

# 에이전트 정의 부분
async def main() -> None:
    try:
        # 여행 정보 에이전트 정의
        travel_agent = AssistantAgent(
            name="travel_agent",
            model_client=OpenAIChatCompletionClient(
                model="gpt-4o-mini"
            ),
            tools=[get_travel_info]
        )

        # 단일 에이전트와 최대 자동 생성 턴 수를 1로 설정하여 팀 정의
        agent_team = RoundRobinGroupChat([travel_agent], max_turns=1)

        print("여행 정보 에이전트 팀이 시작되었습니다.")

        while True:
            # 콘솔에서 사용자 입력 받기
            user_input = input("여행 관련 질문을 입력하세요. (종료하려면 exit): ")
            if user_input.strip().lower() == "exit":
                print("프로그램 종료")
                break

            # 팀 실행 및 콘솔로 메세지 스트리밍
            print(f"사용자 입력: {user_input}")
            stream = agent_team.run_stream(task=user_input)
            await Console(stream)
            print("메세지 스트리밍 완료.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main())