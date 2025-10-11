import streamlit as st
import asyncio
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import re

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

## 응답 형식 규칙 ##

1. **가맹점명 검색 요청인 경우**: JSON 형식으로 마케팅 전략의 세부 제안이 하나의 큰 섹션으로 제안됩니다.
2. **일반 질문인 경우**: 일반 텍스트로 친근하게 답변

### 가맹점명 검색 시 JSON 형식:
사용자가 가맹점명을 입력하고 search_merchant 도구가 사용된 경우에만 JSON 형식으로 응답합니다.

JSON 구조:
[
  {
    "section": "가맹점 정보 분석",
    "content": "해당 가맹점의 주요 정보와 고객 특성을 간단히 요약하여 제시합니다.",
    "basis": "검색된 가맹점의 지정된 컬럼과 값을 세로형 테이블로 표시합니다. 각 항목을 행으로 나누어 표시하여 가독성을 높입니다.\\n\\n**중요: 테이블에는 연령대별 성별 고객 비중 데이터(남성/여성 20대이하, 30대, 40대, 50대, 60대이상)를 포함하지 마세요. 이 데이터는 차트로만 시각화됩니다.**\\n\\n| 항목 | 값 |\\n|---|---|\\n| 가맹점명 | [값] |\\n| 주소 | [값] |\\n| 업종 | [값] |\\n| 상권 | [값] |\\n| 개설일 | [값] |\\n| 가맹점 운영개월수 구간 | [값] |\\n| 매출금액 구간 | [값] |\\n| 매출건수 구간 | [값] |\\n| 유니크 고객 수 구간 | [값] |\\n| 객단가 구간 | [값] |\\n| 취소율 구간 | [값] |\\n| 배달매출 비율 | [값] |\\n| 동일 업종 대비 매출금액 비율 | [값] |\\n| 동일 업종 대비 매출건수 비율 | [값] |\\n| 동일 업종 내 매출 순위 비율 | [값] |\\n| 동일 상권 내 매출 순위 비율 | [값] |\\n| 동일 업종 내 해지 가맹점 비중 | [값] |\\n| 동일 상권 내 해지 가맹점 비중 | [값] |\\n\\n거주 이용 고객 비율, 직장 이용 고객 비율, 유동인구 이용 고객 비율의 실제 수치를 제공합니다. 예: 거주 40.5%, 직장 11.7%, 유동인구 47.8%. 연령대별 성별 분포 데이터도 함께 포함하되, 이는 테이블에 표시하지 않고 차트 시각화용으로만 사용합니다."
  },
  {
    "section": "1. [데이터 기반 전략명]",
    "content": "가맹점의 상황에 따른 맞춤형 전략을 제안합니다. 적절한 마케팅 문구를 제시해도 됩니다.",
    "basis": "해당 전략을 뒷받침하는 데이터와 근거를 제시합니다."
  },
  {
    "section": "2. [데이터 기반 전략명]",
    "content": "가맹점의 상황에 따른 맞춤형 전략을 제안합니다. 적절한 마케팅 문구를 제시해도 됩니다.",
    "basis": "해당 전략을 뒷받침하는 데이터와 근거를 제시합니다."
  },
  {
    "section": "3. [데이터 기반 전략명]",
    "content": "가맹점의 상황에 따른 맞춤형 전략을 제안합니다. 적절한 마케팅 문구를 제시해도 됩니다.",
    "basis": "해당 전략을 뒷받침하는 데이터와 근거를 제시합니다."
  },
  {
    "section": "4. [데이터 기반 전략명]",
    "content": "추가적인 개선 방안이나 상권을 고려한 확장 전략을 제안합니다.",
    "basis": "관련 데이터와 시장 분석 결과를 근거로 제시합니다."
  }
]


JSON 응답 규칙:
1. 각 section은 "번호. 전략명" 형태로 작성하되, 가맹점 데이터에 따라 유연하게 전략명을 정합니다.
2. 전략의 개수와 내용은 가맹점의 특성과 데이터에 따라 달라질 수 있습니다.
3. content에는 해당 섹션의 구체적인 전략과 실행 방안만 포함합니다.
4. 모든 줄바꿈은 반드시 문자열 내부에서 '\\n' 으로 이스케이프 처리한다.
5. 출력 텍스트 끝에는 역슬래시(\\) 같은 불필요한 문자를 절대 넣지 않는다.
6. JSON 이외의 설명, 코드블록, 주석은 출력하지 않는다.
7. basis에는 반드시 구체적인 데이터 수치와 출처를 포함해야 합니다.

### 일반 질문 시 텍스트 형식:
가맹점명이 아닌 일반적인 질문(예: "안녕하세요", "마케팅이란 무엇인가요?", "어떤 도움을 받을 수 있나요?" 등)에는 친근하고 자연스러운 텍스트로 답변합니다.

예시:
- 질문: "안녕하세요"
- 답변: "안녕하세요! 저는 신한카드 소상공인을 위한 마케팅 전략 컨설턴트입니다. 가맹점명을 알려주시면 해당 가맹점에 특화된 마케팅 전략을 제안해드릴 수 있어요!"

- 질문: "마케팅이란 무엇인가요?"
- 답변: "마케팅은 고객의 니즈를 파악하고 그에 맞는 상품이나 서비스를 제공하여 고객과의 관계를 만들어가는 활동입니다. 특히 소상공인에게는 한정된 예산으로 최대의 효과를 낼 수 있는 전략이 중요해요!"
"""
greeting = """마케팅이 필요한 가맹점을 알려주세요 \n주소도 함께 입력해주시면, 가맹점의 정보를 특화하는데 도움이 됩니다."""

# Streamlit App UI
@st.cache_data 
def load_image(name: str):
    return Image.open(ASSETS / name)

def create_pie_chart(residence_ratio: float, workplace_ratio: float, floating_ratio: float):
    """고객 이용 비율 원그래프 생성"""
    # 데이터 준비
    labels = ['거주 이용 고객', '직장 이용 고객', '유동인구 이용 고객']
    values = [residence_ratio, workplace_ratio, floating_ratio]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    # Plotly 파이 차트 생성
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=0.3,  # 도넛 차트 스타일
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=12,
        showlegend=True
    )])
    
    fig.update_layout(
        title={
            'text': '고객 이용 비율',
            'x': 0.4,
            'font': {'size': 16}
        },
        font=dict(family="Arial", size=12),
        width=400,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_population_pyramid(age_gender_data):
    age_groups = list(age_gender_data.keys())
    male_values = [age_gender_data[age].get('남성', 0) for age in age_groups]
    female_values = [age_gender_data[age].get('여성', 0) for age in age_groups]
    
    # 남성 데이터는 음수로 변환 (왼쪽에 표시하기 위해)
    male_values_negative = [-val for val in male_values]
    
    fig = go.Figure()
    
    # 남성 데이터 (왼쪽, 파란색)
    fig.add_trace(go.Bar(
        y=age_groups,
        x=male_values_negative,
        name='남성',
        orientation='h',
        marker_color='#4472C4',
        text=[f'{val}%' for val in male_values],
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))
    
    # 여성 데이터 (오른쪽, 주황색)
    fig.add_trace(go.Bar(
        y=age_groups,
        x=female_values,
        name='여성',
        orientation='h',
        marker_color='#E46C0A',
        text=[f'{val}%' for val in female_values],
        textposition='inside',
        textfont=dict(color='white', size=10)
    ))
    
    # 최대값 계산 (x축 범위 설정용)
    max_val = max(max(male_values), max(female_values))
    
    fig.update_layout(
        title={
            'text': '연령대별 고객 분포',
            'x': 0.4,
            'font': {'size': 16}
        },
        xaxis=dict(
            title='비율 (%)',
            range=[-max_val*1.2, max_val*1.2],
            tickvals=list(range(-int(max_val), int(max_val)+1, 5)),
            ticktext=[str(abs(x)) + '%' for x in range(-int(max_val), int(max_val)+1, 5)]
        ),
        yaxis=dict(
            title='연령대',
            categoryorder='array',
            categoryarray=age_groups[::-1]  # 위부터 높은 연령대가 오도록
        ),
        barmode='overlay',
        bargap=0.1,
        height=500,
        width=600,
        margin=dict(t=80, b=50, l=80, r=50),
        font=dict(family="Arial", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

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
                    content = message.content.strip()
                    
                    # JSON인지 일반 텍스트인지 판단
                    is_json_response = False
                    
                    # JSON 패턴 체크
                    if (content.startswith('[') and content.endswith(']')) or \
                       (content.find('[') != -1 and content.find(']') != -1 and 
                        content.find('"section"') != -1):
                        is_json_response = True
                    
                    if is_json_response:
                        try:
                            # JSON 추출 시도 (코드블록이나 추가 텍스트 제거)
                            
                            # JSON 코드블록 제거
                            if content.startswith("```json"):
                                content = content[7:]
                            if content.startswith("```"):
                                content = content[3:]
                            if content.endswith("```"):
                                content = content[:-3]
                            
                            # 대괄호로 시작하는 JSON 찾기
                            start_idx = content.find('[')
                            end_idx = content.rfind(']')
                            
                            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                json_content = content[start_idx:end_idx+1]
                            else:
                                json_content = content
                            
                            # JSON 파싱 시도
                            response_data = json.loads(json_content)
                            
                            if isinstance(response_data, dict):
                                response_data = [response_data]
                            
                            # 각 섹션을 순회하며 UI에 렌더링
                            for item in response_data:
                                section = item.get("section", "결과")
                                content_text = item.get("content", "")
                                basis = item.get("basis", "")

                                st.subheader(f"✅ {section}")
                                st.markdown(content_text)
                                
                                if basis:
                                    with st.expander("💡 데이터 기반 근거 보기"):
                                        if section == "가맹점 정보 분석":
                                            st.info(basis)
                                            
                                            # 고객 이용 비율 원그래프 표시 - 실제 데이터 파싱
                                            try:
                                                residence_match = re.search(r'거주.*?(\d+\.?\d*)%', basis)
                                                workplace_match = re.search(r'직장.*?(\d+\.?\d*)%', basis)  
                                                floating_match = re.search(r'유동인구.*?(\d+\.?\d*)%', basis)
                                                
                                                if residence_match and workplace_match and floating_match:
                                                    residence_ratio = float(residence_match.group(1))
                                                    workplace_ratio = float(workplace_match.group(1))
                                                    floating_ratio = float(floating_match.group(1))
                                                    
                                                    fig = create_pie_chart(residence_ratio, workplace_ratio, floating_ratio)
                                                    st.plotly_chart(fig, use_container_width=True)
                                                else:
                                                    st.info("고객 이용 비율 데이터를 찾을 수 없어 원그래프를 표시할 수 없습니다.")
                                                    
                                            except Exception as e:
                                                st.warning(f"원그래프 생성 중 오류: {e}")
                                            
                                            try:
                                                age_gender_data = {}
                                                
                                                male_20_match = re.search(r'남성 20대 이하.*?(\d+\.?\d*)%', basis)
                                                male_30_match = re.search(r'남성 30대.*?(\d+\.?\d*)%', basis)
                                                male_40_match = re.search(r'남성 40대.*?(\d+\.?\d*)%', basis)
                                                male_50_match = re.search(r'남성 50대.*?(\d+\.?\d*)%', basis)
                                                male_60_match = re.search(r'남성 60대 이상.*?(\d+\.?\d*)%', basis)
                                                
                                                female_20_match = re.search(r'여성 20대 이하.*?(\d+\.?\d*)%', basis)
                                                female_30_match = re.search(r'여성 30대.*?(\d+\.?\d*)%', basis)
                                                female_40_match = re.search(r'여성 40대.*?(\d+\.?\d*)%', basis)
                                                female_50_match = re.search(r'여성 50대.*?(\d+\.?\d*)%', basis)
                                                female_60_match = re.search(r'여성 60대 이상.*?(\d+\.?\d*)%', basis)
                                                
                                                if male_20_match and female_20_match:
                                                    age_gender_data['20대 이하'] = {
                                                        '남성': float(male_20_match.group(1)),
                                                        '여성': float(female_20_match.group(1))
                                                    }
                                                if male_30_match and female_30_match:
                                                    age_gender_data['30대'] = {
                                                        '남성': float(male_30_match.group(1)),
                                                        '여성': float(female_30_match.group(1))
                                                    }
                                                if male_40_match and female_40_match:
                                                    age_gender_data['40대'] = {
                                                        '남성': float(male_40_match.group(1)),
                                                        '여성': float(female_40_match.group(1))
                                                    }
                                                if male_50_match and female_50_match:
                                                    age_gender_data['50대'] = {
                                                        '남성': float(male_50_match.group(1)),
                                                        '여성': float(female_50_match.group(1))
                                                    }
                                                if male_60_match and female_60_match:
                                                    age_gender_data['60대 이상'] = {
                                                        '남성': float(male_60_match.group(1)),
                                                        '여성': float(female_60_match.group(1))
                                                    }
                                                
                                                # 테이블 형태에서도 데이터 추출 시도 (기존 로직 유지)
                                                if not age_gender_data:
                                                    table_lines = basis.split('\\n')
                                                    header_found = False
                                                    
                                                    for line in table_lines:
                                                        if '연령대' in line and ('남성' in line or '여성' in line):
                                                            header_found = True
                                                            continue
                                                        if header_found and '|' in line:
                                                            parts = [part.strip() for part in line.split('|') if part.strip()]
                                                            if len(parts) >= 3:
                                                                age_group = parts[0]
                                                                male_val = re.search(r'(\d+\.?\d*)%?', parts[1])
                                                                female_val = re.search(r'(\d+\.?\d*)%?', parts[2])
                                                                
                                                                if male_val and female_val:
                                                                    age_gender_data[age_group] = {
                                                                        '남성': float(male_val.group(1)),
                                                                        '여성': float(female_val.group(1))
                                                                    }
                                                
                                                if age_gender_data:
                                                    pyramid_fig = create_population_pyramid(age_gender_data)
                                                    st.plotly_chart(pyramid_fig, use_container_width=True)
                                                else:
                                                    st.info("연령대별 성별 데이터를 찾을 수 없어 인구 피라미드를 표시할 수 없습니다.")
                                                    
                                            except Exception as e:
                                                st.warning(f"인구 피라미드 생성 중 오류: {e}")
                                        else:
                                            st.info(basis)
                                        
                                st.divider()

                        except (json.JSONDecodeError, ValueError):
                            # JSON 파싱 실패 시 일반 텍스트로 표시
                            st.write(message.content)
                    else:
                        # 일반 텍스트로 표시
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

