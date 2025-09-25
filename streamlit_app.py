import streamlit as st
import asyncio
import json

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from PIL import Image
from pathlib import Path

# 환경변수
ASSETS = Path("assets")
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

system_prompt = """
당신은 데이터를 기반으로 마케팅 전략을 제안하는 전문 컨설턴트입니다.
당신의 답변은 반드시 JSON 형식이어야 합니다. 다른 설명 없이 JSON 코드만 출력해야 합니다.

JSON 구조는 다음과 같은 객체들의 리스트(배열) 형태여야 합니다:
[
  {
    "section": "마케팅 전략의 주제나 단계",
    "content": "해당 주제에 대한 구체적인 마케팅 제안 내용. 만약 내용에 마크다운 테이블과 같이 여러 줄이 포함된다면, 반드시 줄바꿈을 '\\n' 으로 이스케이프 처리해야 합니다.",
    "basis": "해당 제안을 뒷받침하는 데이터 기반의 근거"
  }
]

## 예시 ##
다음은 content에 마크다운 테이블이 포함된 경우의 올바른 JSON 형식입니다:
[
  {
    "section": "주요 고객 분석",
    "content": "아래는 주요 고객층의 연령 및 성별 분포입니다.\\n\\n| 연령대 | 남성 | 여성 |\\n|---|---|---|\\n| 20대 | 15% | 25% |\\n| 30대 | 30% | 20% |\\n| 40대 | 5% | 5% |",
    "basis": "가맹점의 최근 3개월 결제 데이터를 분석한 결과입니다."
  }
]

규칙:
1. 모든 줄바꿈은 반드시 문자열 내부에서 '\\n' 으로 이스케이프 처리한다.
2. 출력 텍스트 끝에는 역슬래시(\\) 같은 불필요한 문자를 절대 넣지 않는다.
3. JSON 이외의 설명, 코드블록, 주석은 출력하지 않는다.
4. 모든 제안(content)에는 반드시 데이터에 기반한 근거(basis)가 함께 제시되어야 한다.
"""
greeting = """마케팅이 필요한 가맹점을 알려주세요 \n주소도 함께 입력해주시면, 가맹점의 정보를 특화하는데 도움이 됩니다."""

# Streamlit App UI
@st.cache_data 
def load_image(name: str):
    return Image.open(ASSETS / name)

st.set_page_config(page_title="2025년 빅콘테스트 AI데이터 활용분야 - SAVAGE")

def clear_chat_history():
    st.session_state.messages = [SystemMessage(content=system_prompt), AIMessage(content=greeting)]

# 사이드바
with st.sidebar:
    st.image(load_image("shc_ci_basic_00.png"), width='stretch')
    st.markdown("<p style='text-align: center;'>2025 Big Contest</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI DATA 활용분야</p>", unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1,2,1])  # 비율 조정 가능
    with col2:
        st.button('Clear Chat History', on_click=clear_chat_history)

# 헤더 컨테이너
header_container = st.container()
with header_container:
    st.title("신한카드 소상공인 🔑 비밀상담소")
    st.subheader("#우리동네 #숨은맛집 #소상공인 #마케팅 #전략 .. 🤤")
    st.write("")

# 메시지 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt),
        AIMessage(content=greeting)
    ]

chat_container = st.container()

def render_messages():
    """모든 메시지를 렌더링하는 함수"""
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if isinstance(message, SystemMessage):
                continue
            elif isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    try:
                        # JSON 파싱 시도
                        response_data = json.loads(message.content)
                        
                        if isinstance(response_data, dict):
                            response_data = [response_data]
                        
                        # 각 섹션을 순회하며 UI에 렌더링
                        for item in response_data:
                            section = item.get("section", "결과")
                            content = item.get("content", "")
                            basis = item.get("basis", "")

                            st.subheader(f"✅ {section}")
                            st.markdown(content)
                            
                            if basis:
                                with st.expander("💡 데이터 기반 근거 보기"):
                                    st.info(basis)
                            st.divider()

                    except json.JSONDecodeError:
                        st.write(message.content)

# 초기 메시지 렌더링
render_messages()

def render_chat_message(role: str, content: str):
    with st.chat_message(role):
        st.markdown(content.replace("<br>", "  \n"))

# LLM 모델 선택
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # 최신 Gemini 2.5 Flash 모델
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )

# MCP 서버 파라미터(환경에 맞게 명령 수정)
server_params = StdioServerParameters(
    command="uv",
    args=["run","mcp_server.py"],
    env=None
)

# 사용자 입력 처리
async def process_user_input():
    """사용자 입력을 처리하는 async 함수"""
    async with stdio_client(server_params) as (read, write):
        # 스트림으로 ClientSession을 만들고
        async with ClientSession(read, write) as session:
            # 세션을 initialize 한다
            await session.initialize()

            # MCP 툴 로드
            tools = await load_mcp_tools(session)

            # 에이전트 생성
            agent = create_react_agent(llm, tools)

            # 에이전트에 전체 대화 히스토리 전달
            agent_response = await agent.ainvoke({"messages": st.session_state.messages})
            
            # AI 응답을 대화 히스토리에 추가
            ai_message = agent_response["messages"][-1]  # 마지막 메시지가 AI 응답

            return ai_message.content
            

input_container = st.container()
with input_container:
    if query := st.chat_input("가맹점 이름을 입력하세요"):
        # 사용자 메시지 추가
        st.session_state.messages.append(HumanMessage(content=query))
        st.rerun() # 페이지 새로고침하여 새로운 메시지를 표시

# 사용자 입력이 있을 때는 채팅 히스토리에 추가하여 UI 유지
if len(st.session_state.messages) > 2:  
    last_message = st.session_state.messages[-1]
    if isinstance(last_message, HumanMessage):
        with st.spinner("Thinking..."):
            try:
                reply = asyncio.run(process_user_input())
                st.session_state.messages.append(AIMessage(content=reply))
                # 메시지 추가 후 다시 렌더링
                render_messages()
                st.rerun()  # 새로운 응답을 표시하기 위해 페이지 새로고침
                
            except* Exception as eg:
                for i, exc in enumerate(eg.exceptions, 1):
                    error_msg = f"오류가 발생했습니다 #{i}: {exc!r}"
                    st.session_state.messages.append(AIMessage(content=error_msg))
                render_messages()
                st.rerun()

