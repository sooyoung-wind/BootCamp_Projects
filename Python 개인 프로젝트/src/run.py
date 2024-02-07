# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:43:22 2023

@author: Soo.Y
"""
from src.GAEM_WORD import gameRun
from src.search import my_search
from src.brown_smog import call_smog
from src.weather import call_weather_info
from src.my_speech import MySpeech
from src.common.load_modules import load_all_modules
from src.common.extracting_search_words import extracting_search_words
from src.common.user_login import doingLogin
from src.sql.SQL_handing import *
import os


def run():
    isLoding = load_all_modules()  # call all of modules
    if isLoding == True:
        print("=================================")
        print("시스템 로드를 완료했습니다.")
        print("=================================")
        isDoing = True
    else:
        print("=================================")
        print("시스템 로드를 완료하지 못했습니다.")
        print("=================================")
        isDoing = False

    # 유저 정보 입력
    if not os.path.isfile("DB/siri_userData.db"):
        print("사용자 정보가 없습니다.")
        sql_create() # DB 생성
        sql_new_user() # 유저 입력

    while isLoding:
        siri = MySpeech()
        siri.speak()

        siri = doingLogin(siri)

        siri.text = f"{siri.userInfo['name']}님 반갑습니다. 저는 {siri.userInfo['call']}입니다. 이제부터 저를 불러주시면 됩니다."
        siri.speak()

        isDoing = True
        while isDoing:
            # stt tts 구현
            siri.voice_to_text()

            if siri.userInfo['call'] in siri.text:
                print('호출 완료')
                siri.text = "찾으셨습니까? 제가 무엇을 할까요?"
                siri.speak()
                siri.voice_to_text()

                if "구글" in siri.text:
                    search_query = extracting_search_words(siri.text)
                    full_url = "https://www.google.com/search?q=" + search_query
                    if hasattr(siri, 'driver'):
                        my_search(full_url, siri)
                    else:
                        siri.driver = my_search(full_url, siri)

                if "네이버" in siri.text:
                    search_query = extracting_search_words(siri.text)
                    full_url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=" + search_query
                    if hasattr(siri, 'driver'):
                        my_search(full_url, siri)
                    else:
                        siri.driver = my_search(full_url, siri)
                if "미세먼지" in siri.text:
                    location_name = extracting_search_words(siri.text)
                    call_smog(location_name, siri)
                if "날씨" in siri.text:
                    location_name = extracting_search_words(siri.text)
                    call_weather_info(location_name, siri)
                if "기능" in siri.text:
                    siri.text = "현재 구현된 기능은 검색기능, 날씨, 미세먼지 조회가 있습니다."
                    siri.speak()
                if "게임" in siri.text:
                    gameRun()

            if "로그아웃" in siri.text:
                isDoing = False

            if "종료" in siri.text:
                siri.text = "종료합니다."
                siri.speak()
                isDoing = False
                isLoding = False
