import asyncio
import pandas as pd

from langchain_experimental.tools.python.tool import PythonAstREPLTool
from autogen_ext.tools.langchain import LangChainToolAdapter
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from dotenv import load_dotenv

load_dotenv()

async def main():
    df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv")
    tool = LangChainToolAdapter(PythonAstREPLTool(locals={"df":df}))
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent=AssistantAgent(
        "assistant",
        tools=[tool],
        model_client=model_client,
        system_message="데이터셋에 액세스하려면 'df' 변수를 사용하세요."
    )
    await Console(
        agent.on_messages_stream(
            [TextMessage(content="탑승객의 평균 나이?", source="user")], CancellationToken()
        )
    )

asyncio.run(main())