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
    "content": "해당 주제에 대한 구체적인 마케팅 제안 내용. 만약 내용에 마크다운 테이블과 같이 여러 줄이 포함된다면, 반드시 개행 문자를 '\\n'으로 이스케이프 처리해야 합니다.",
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

사용자가 가맹점명을 입력하면, 당신은 제공된 데이터를 분석하여 위 JSON 형식에 맞춰 체계적인 마케팅 전략을 제안해야 합니다.
모든 제안(content)에는 반드시 데이터에 기반한 근거(basis)가 함께 제시되어야 합니다.
분석 결과는 가능한 표(마크다운 형식)를 사용하여 'content'에 포함시키면 가독성을 높일 수 있습니다.
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

# 헤더
st.title("신한카드 소상공인 🔑 비밀상담소")
st.subheader("#우리동네 #숨은맛집 #소상공인 #마케팅 #전략 .. 🤤")
st.image(load_image("image_gen3.png"), width='stretch', caption="🌀 머리아픈 마케팅 📊 어떻게 하면 좋을까?")
st.write("")

# 메시지 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt),
        AIMessage(content=greeting)
    ]

# 초기 메시지 화면 표시
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

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
            

# 사용자 입력 창
if query := st.chat_input("가맹점 이름을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.messages.append(HumanMessage(content=query))
    render_chat_message("user", query)

    with st.spinner("Thinking..."):
        try:
            # 사용자 입력 처리
            reply = asyncio.run(process_user_input())
            st.session_state.messages.append(AIMessage(content=reply))
            with st.chat_message("assistant"):
                try:
                    # JSON 파싱
                    response_data = json.loads(reply)
                    
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
                    # LLM이 유효한 JSON을 생성하지 못한 경우, 원본 텍스트를 그대로 보여줌
                    st.error("결과를 분석하는 데 문제가 발생했습니다. 원본 메시지를 확인해주세요.")
                    st.markdown(reply)
        except* Exception as eg:
            # 오류 처리
            for i, exc in enumerate(eg.exceptions, 1):
                error_msg = f"오류가 발생했습니다 #{i}: {exc!r}"
                st.session_state.messages.append(AIMessage(content=error_msg))
                render_chat_message("assistant", error_msg)
