# Ref: https://www.youtube.com/watch?v=8mzveVswcDY
import datetime
import os
import autogen
import dotenv
from pathlib import Path
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.code_utils import create_virtual_env

dotenv.load_dotenv()

# 소스코드 실행기 만들기
work_dir = Path('autostock')
virtual_env_context = create_virtual_env(Path(work_dir, '.venv'))
executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir=work_dir,
    virtual_env_context=virtual_env_context
)

# 에이전트 만들기
code_executor_agent = autogen.UserProxyAgent(   # 비서 에이전트
    name='code_executor_agent',
    llm_config=False,
    code_execution_config={'executor':executor},
    human_input_mode='ALWAYS',
    default_auto_reply='하던 일을 계속해주세요. 모든 것을 마쳤다면, TERMINATE 라고 대답해주세요.'
)
code_writer_agent = autogen.AssistantAgent(     # 개발자 에이전트
    name='code_writer_agent',
    llm_config={'model':'gpt-4o-mini', 'api_key':os.environ['OPENAI_API_KEY']},
    code_execution_config=False,
    human_input_mode='NEVER'
)

# 대화 시작하기
today = datetime.datetime.now().date()
message = (
    f'오늘의 날짜는 {today}야.'
    '삼성전자(KRX: 005930) and SK하이닉스(KRX: 000660)의 주가의 연초 대비 증감률(YTD)를 시각화하는 그래프를 만들어줘.'
    '너가 작성한 소스코드는 꼭 마크다운 블록 ``` 에 들어 있어야 해.'
    '작성한 그래프는 `ytd_stock_gains.png`라는 이름의 파일로 저장하도록 해.'
)
chat_result = code_executor_agent.initiate_chat(    # 개발자 에이전트에게 메세지를 담아서 전달
    recipient=code_writer_agent,
    message=message
)
