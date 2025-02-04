import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import StateGraph, END
import functools
from typing import Annotated, Sequence, TypedDict
import operator

# Initialize environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
os.environ['TAVILY_API_KEY'] = os.environ.get("TAVILY_API_KEY")

# Initialize the language model (LLM)
llm = ChatOpenAI(model="gpt-4o-mini")

# Create Agent function
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Tools: Tavily Search and Python REPL
tavily_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPLTool()

# Load documents and create embeddings
loader = DirectoryLoader('./source', glob="./*.txt", loader_cls=TextLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10, length_function=len)
new_docs = text_splitter.split_documents(documents=docs)

# BGE Embeddings
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

# Create Retriever using Vector DB
db = Chroma.from_documents(new_docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

# Perfume Chat Tool: General Perfume-Related Chat Companion
@tool
def general_perfume_related_chat_companion(state):
    """Use this to engage in a general perfume-related chat."""
    print('-> Calling Perfume Chat ->')
    
    # Ensure `state` contains a valid 'user_input' field
    user_input = state
    if not isinstance(user_input, str):
        print("Invalid input format. Expected a string.")
        return {"output": "Invalid input format. Expected a string."}

    print('User Input:', user_input)
    
    # Basic responses based on user input
    response_map = {
        "안녕하세요!": "안녕하세요! 오늘 어떤 향수를 찾으시나요? 특별한 날을 위한 향수가 필요하신가요?",
        "힘든 하루였어요.": "오늘 정말 고생하셨네요. 따뜻하고 위로가 되는 향수를 추천드릴게요. 바닐라나 앰버 계열의 향은 마음을 편안하게 만들어줍니다.",
        "세계에서 가장 비싼 향수를 추천해 주세요!": "세계에서 가장 비싼 향수요? '클라이브 크리스찬 No.1 임페리얼 마제스티'가 떠오르네요. 한 병에 억 단위라니, 어울리는 향수를 찾는 것도 럭셔리하네요! 더 실용적인 선택도 필요하시면 알려주세요.",
        "사막이 연상되는 향수가 있나요?": "사막을 연상시키는 향수를 찾으신다면, '소바쥬'는 사막의 공기를 담은 시트러스와 우디 계열로 완벽합니다. 또한 '엉브레'는 사막의 빛나는 모래를 떠올리게 하는 앰버 계열입니다.",
        "오늘 날씨는 어때요?": "날씨 정보는 제공하지 않지만, 맑은 날씨라면 시트러스 계열 향수가 어울릴 것 같아요. '조 말론 라임 바질 앤 만다린'을 추천드립니다!",
        "요즘 몸이 안 좋은데 병원을 좀 추천해줘": "요즘 몸 상태가 좋지 않으시다니 걱정되네요. 병원 추천은 제가 드릴 수 없지만, 따뜻하고 안정감을 주는 향기로 몸과 마음을 조금이라도 편안하게 만들어보는 건 어떨까요? 예를 들어, 라벤더나 카모마일 노트가 포함된 향수는 안정감을 주는 데 도움을 줄 수 있습니다. 무엇보다 건강이 우선이니 꼭 가까운 병원에서 진료를 받아보세요!",
        "나 오늘 기분이 너무 안 좋은데 어떻게 해야 할까?": "기분이 안 좋을 때는 마음을 다독여줄 향을 맡아보는 것은 어떠실까요? 따듯하고 부드러운 향이 당신을 감싸고 하루를 활기차게 만들어줄 거예요. 잠시 산책을 즐기는 것도 도움이 될 거예요!"
    }

    # Default response
    response = response_map.get(user_input, "저는 향수와 관련된 이야기를 나누는 챗봇이에요! 원하는 향수를 찾거나, 기분에 맞는 향수를 추천해 드릴게요. 어떤 향수를 찾고 계신가요?")
    
    return {"output": response}

# RAG Tool: Python function -> tool
@tool
def RAG(state):
    """Use this to execute RAG. If the question is related to Japan or Sports, retrieve the results."""
    print('-> Calling RAG ->')
    
    # Ensure `state` contains a valid 'question' field and extract it as a string
    question = state  # Make sure 'question' exists and is a string
    if not isinstance(question, str):
        print("Invalid question format. Expected a string.")
        return {"output": "Invalid question format. Expected a string."}

    print('Question:', question)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = retrieval_chain.invoke(question)
    return result


# Agent Node: Takes state, agent, and name as input and returns message as HumanMessage
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Supervisor Chain Creation
members = ["RAG", "Researcher", "Coder", "GeneralPerfumeRelatedChatCompanion"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Use RAG tool when questions"
    " are related to Japan or Sports. Use GeneralPerfumeRelatedChatCompanion tool when the question"
    " is related to perfume or general chat. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
options = ["FINISH"] + members
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [{"enum": options}],
            }
        },
        "required": ["next"],
    },
}

# Define supervisor chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# Create workflow with nodes
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

research_agent = create_agent(llm, [tavily_tool], "You are a web researcher.")
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code to analyze data and generate charts using matplotlib.",
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

RAG_agent = create_agent(
    llm,
    [RAG],
    "Use this tool when questions are related to Japan or Sports.",
)
rag_node = functools.partial(agent_node, agent=RAG_agent, name="RAG")

general_perfume_related_chat_companion_agent = create_agent(
    llm,
    [general_perfume_related_chat_companion],  # Use the perfume chat tool
    "당신은 사용자와 친근하게 대화하는 센티크 챗봇입니다. 사용자의 입력을 이해하고 적절히 답변합니다."
)
general_perfume_related_chat_companion_node = functools.partial(agent_node, agent=general_perfume_related_chat_companion_agent, name="GeneralPerfumeRelatedChatCompanion")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("RAG", rag_node)
workflow.add_node("supervisor", supervisor_chain)
workflow.add_node("GeneralPerfumeRelatedChatCompanion", general_perfume_related_chat_companion_node)

# Create edges between nodes
for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Set entry point
workflow.set_entry_point("supervisor")
graph = workflow.compile()

# Run the workflow with some example messages
def run_workflow():
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(content="겨울에는 어떤 향수가 인기가 많아?")
            ]
        }
    ):
        if "__end__" not in s:
            print(s)
            print("----")

if __name__ == "__main__":
    run_workflow()
