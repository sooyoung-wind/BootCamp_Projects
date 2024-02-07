from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from openai import OpenAI
import streamlit as st
import time
import numpy as np
from src.tts import speak
from src.stt import SttEngine
import pandas as pd
from langchain_core.messages import HumanMessage
import re

check = pd.read_table("./check_first.txt")

if check.section[0] == 0:
    speak('안녕하세요. 로딩중 입니다.', False)
    check.section[0] += 1
    check.to_csv("./check_first.txt", sep='\t', index=False)

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
google_api_key = os.environ['GOOGLE_API_KEY']
gemini = ChatGoogleGenerativeAI(
    model='gemini-pro', convert_system_message_to_human=True, temperature=0)
vision = ChatGoogleGenerativeAI(
    model="gemini-pro-vision", convert_system_message_to_human=True, temperature=0)

##############################
### section 1
### 초기
##############################

if check.section[0] == 1:
    st.header("나 대신 발표하는 AI 봇")
    speak('오늘 발표할 AI 봇입니다. 반갑습니다.')
    speak('아직 배워야할 것들이 많은 AI 봇입니다. 그래서 다소 어눌한 말투라도 양해바랍니다. ')
    check.section[0] += 1
    check.to_csv("./check_first.txt", sep='\t', index=False)
    st.write("----")

##############################
### section 2
### 소개
##############################

if check.section[0] == 2:
    st.header('주제 : 뉴스 Q&A')
    image_file = './fig/news_QandA.png'
    st.image(image_file, use_column_width=True, width=200)
    
    with st.spinner('검색 중...'):
        send_msg =  '''발표를 하고자 한다. 발표의 도입부분의 대본을 작성해줘.
                    발표할 주제는 "뉴스 Q&A"이다. 뉴스 Q&A는 수집된 뉴스를 토대로
                    유저가 질문을 하면 대답을 해주는 AI를 만드는 주제이다. 
                    발표자는 미연, 수영, 은채, 3명이고, 팀명은 MSE이다. 사용된 모델은 gemini를 사용했다.
                    다음과 같은 순서로 발표를 해줘.
                    1. 발표자 및 팀명
                    2. 발표할 주제 및 설명
                    3. 사용한 모델
                    위의 제목은 언급하지 않는다.
                    '''
        result = gemini.invoke(send_msg)
    
    with st.spinner('대답 중...'):
        result = result.content
        paragraphs = result.split('\n')
        for paragraph in paragraphs:
            if paragraph:
                st.write(paragraph)
                paragraph = re.sub(r'\d+\.', '', paragraph)
                paragraph = paragraph.replace('*', '')
                paragraph = re.sub(r'\([^)]*\)', '', paragraph)
                speak(paragraph, False)
        
    
    check.section[0] += 1
    check.to_csv("./check_first.txt", sep='\t', index=False)
    st.write("----")

##############################
### section 3
### RAG 이란
##############################

if check.section[0] == 3:
    st.header('RAG(Retrieval-Augmented Generation)')
    
    with st.spinner('대답 중...'):
        speak('LLM모델을 직접 학습할 수 없기 때문에 RAG를 사용해서 특정 뉴스에 대한 질의가 가능하도록 구현했습니다.')
        speak('RAG에 대해서 설명해드리겠습니다.')
    
    with st.spinner('검색 중...'):
        send_msg = '''Retrieval-Augmented Generation에
                    대해서 아주 간단하게 설명해주세요. 2문장으로 해줘.
                    (한국어로 설명해줘야 한다.)'''
        result = gemini.invoke(send_msg)
    
    with st.spinner('대답 중...'):
        result = result.content
        paragraphs = result.split('\n')
        for paragraph in paragraphs:
            if paragraph:
                st.write(paragraph)
                paragraph = re.sub(r'\d+\.', '', paragraph)
                paragraph = paragraph.replace('*', '')
                paragraph = re.sub(r'\([^)]*\)', '', paragraph)
                speak(paragraph, False)
        
    check.section[0] += 1
    check.to_csv("./check_first.txt", sep='\t', index=False)
    st.write("----")
    
##############################
# section 4
# RAG 그림 설명
##############################
if check.section[0] == 4:
    with st.spinner('대답 중...'):
        speak('다음 그림은 RAG 논문에서 가져온 그림입니다.')
    
    image_file = './fig/RAG.png'
    st.image(image_file, use_column_width=True, width=200)
    
    with st.spinner('음성 준비중...'):
        image_url = './fig/RAG.png'
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "이 그림에 대해서 설명해주세요. 한국어로 설명해줘야 한다. 상세하게 설명해주면 좋겠습니다.",
                },  # You can optionally provide text parts
                {"type": "image_url", "image_url": image_url},
            ]
        )
        result = vision.invoke([message])
        result = result.content
    
    # '\n' 문자를 기준으로 텍스트를 분할
    with st.spinner('대답 중...'):
        paragraphs = result.split('\n')
        for paragraph in paragraphs:
            if paragraph:
                st.write(paragraph)
                paragraph = re.sub(r'\d+\.', '', paragraph)
                paragraph = paragraph.replace('*', '')
                paragraph = re.sub(r'\([^)]*\)', '', paragraph)
                speak(paragraph, False)
    
    check.section[0] += 1
    check.to_csv("./check_first.txt", sep='\t', index=False)
    st.write("----")

##############################
# section 5
# 시연 준비 단계
##############################
if check.section[0] == 5:
    # 페이지 제목
    st.header("뉴스 Q&A 챗봇을 시연해보겠습니다.")
    speak("뉴스 Q&A 챗봇을 시연해보겠습니다.", False)
    st.write("제공된 뉴스정보만 제공합니다. 아직 네이버 뉴스 api 연동은 안되어 있습니다.")
    check.section[0] += 1
    check.to_csv("./check_first.txt", sep='\t', index=False)

##############################
### section 6
### 시연 시작
##############################
if check.section[0] == 6:
    client = OpenAI(api_key=API_KEY)

    # thread id를 하나로 관리하기 위함
    if 'thread_id' not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id

    # thread_id, assistant_id 설정
    thread_id = st.session_state.thread_id
    # 미리 만들어 둔 Assistant
    assistant_id = "asst_NNd0u67Kj8WpIeNnhgYwXX2D"
    
    prompt = st.chat_input("궁금한 점을 물어봐주세요. 예) 오늘의 뉴스는?")

    # 메세지 모두 불러오기
    thread_messages = client.beta.threads.messages.list(thread_id, order="asc")
    
    # 메세지 역순으로 가져와서 UI에 뿌려주기
    for msg in thread_messages.data:
        with st.chat_message(msg.role):
            st.write(msg.content[0].text.value)

    if prompt:
        if prompt == "next section":
            word = "시연을 마무리하고 다음으로 넘어갑니다."
            speak(word)
            check.section[0] += 1
            check.to_csv("./check_first.txt", sep='\t', index=False)

        if prompt != "next section":
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=prompt
            )

            # 입력한 메세지 UI에 표시
            with st.chat_message(message.role):
                st.write(message.content[0].text.value)

            # RUN을 돌리는 과정
            run = client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
            )

            with st.spinner('응답 기다리는 중...'):
                # RUN이 completed 되었나 1초마다 체크
                speak('잠시만 기다려주세요.')
                while run.status != "completed":
                    time.sleep(1)
                    run = client.beta.threads.runs.retrieve(
                        thread_id=thread_id,
                        run_id=run.id
                    )

            # while문을 빠져나왔다는 것은 완료됐다는 것이니 메세지 불러오기
            messages = client.beta.threads.messages.list(
                thread_id=thread_id
            )
            # 마지막 메세지 UI에 추가하기
            with st.chat_message(messages.data[0].role):
                st.write(messages.data[0].content[0].text.value)
                ### speaking code
                # speak(messages.data[0].content[0].text.value, False)

##############################
### section 7
### 시연 준비 단계
# ##############################
if check.section[0] == 7:
    st.header('질의 결과에 대한 문제점')
    image_file = './fig/comparing_results.png'
    st.image(image_file, use_column_width=True, width=200)

    with st.spinner('대답 중...'):
        speak('위의 그림은 오늘의 뉴스를 질의했을 때의 답변입니다.')
        speak('왼쪽 그림이 오늘의 뉴스에서 물음표를 추가해서 질의한 결과이고 오른쪽 그림이 오늘의 뉴스에서 물음표를 제외해서 질의한 결과입니다.')
        speak('오늘의 뉴스에 물음표를 붙이면 사전에 입력된 프롬프트에 의해 검색이 나왔으나, 물음표가 없으면 관련된 정보가 없다고 나왔습니다.')
        speak('저희는 물음표가 없으면 단어 그 자체를 검색한 것이 아닐까 생각해봅니다.')
        speak('그래서 추후 실험을 통해 프롬프트를 개선해야 할 부분이라고 생각합니다.')

    check.section[0] += 1
    check.to_csv("./check_first.txt", sep='\t', index=False)
    st.write("----")

#############################
### section 8
### 마무리
##############################
if check.section[0] == 8:
    st.header('지금까지 2조 발표였습니다. 경청해 주셔서 감사합니다.')
    speak('지금까지 2조 발표였습니다. 경청해 주셔서 감사합니다.', False)

    check.section[0] += 1
    check.to_csv("./check_first.txt", sep='\t', index=False)
    st.write("----")


