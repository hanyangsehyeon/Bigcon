# KAIT 주최 및 신한카드 주관 - 2025 빅콘테스트 AI데이터 활용분야
### "내 가게를 살리는 AI 비밀상담사 - 가맹점별 찰떡 마케팅 전략을 찾아라"

<br>

## 서비스 웹 페이지
[https://bigconsavage.streamlit.app/](https://bigconsavage.streamlit.app/)

<br>

## 👩‍💻👨‍💻 Team 세비지
|김세희|김지원|이지윤|전세현|
|:---:|:---:|:---:|:---:|
|프롬프팅, UI|mcp툴|데이터분석|mcp툴, 데이터분석|
|[sehee0207](https://github.com/sehee0207)|[jiwonniddaaa](https://github.com/jiwonniddaaa)|[jiyunni](https://github.com/jiyunni)|[hanyangsehyeon](https://github.com/hanyangsehyeon)|
|![](https://avatars.githubusercontent.com/u/65457903?v=4)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|![](https://avatars.githubusercontent.com/u/120795436?v=4)|![](https://avatars.githubusercontent.com/u/146607256?v=4)|![](https://avatars.githubusercontent.com/u/218353569?v=4)|

<br>


## 로컬 구동 환경 구성

```bash
# On macOS and Linux.
# git소스 복사하기
git clone https://github.com/thjeong/shcard_2025_bigcontest
cd shcard_2025_bigcontest

# venv 환경 설정 (사전 uv 설치 필요)
uv venv
source .venv/bin/activate

# 필요한 python library 설치
uv pip install -r requirements.txt

# streamlit 환경 변수 저장용 폴더 생성 + GOOGLE_API_KEY환경 변수 파일 생성
# (Google API KEY)는 생성 필요
mkdir .streamlit
echo 'GOOGLE_API_KEY="(Google API KEY)"' > .streamlit/secrets.toml

# 로컬에서 실행
uv run streamlit run streamlit_app.py
```

```bat
:: On Windows
:: git 소스 복사하기
git clone https://github.com/thjeong/shcard_2025_bigcontest
cd shcard_2025_bigcontest

:: venv 환경 설정 (사전 uv 설치 필요)
uv venv
call .venv\Scripts\activate.bat

:: 필요한 python library 설치
uv pip install -r requirements.txt

# streamlit 환경 변수 저장용 폴더 생성 + GOOGLE_API_KEY환경 변수 파일 생성
# (Google API KEY)는 생성 필요
mkdir .streamlit
echo GOOGLE_API_KEY="(Google API KEY)" > .streamlit\secrets.toml

:: 로컬에서 실행
uv run streamlit run streamlit_app.py
```
