# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:43:34 2023

@author: Soo.Y
"""
import requests
import json
from datetime import datetime
from src.common.change_raw_to_dict import change_raw_to_dict


def call_smog(target_name: str, myDriver):
    '''
    Parameters
    ----------
    target_name : str
        DESCRIPTION.
    airKorea_api_key : str
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    '''
    if target_name in ['전국', '서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '세종']:
        pass
    else:
        myDriver.text = "지역명이 잘못되었습니다."
        myDriver.speak()
        # siri.text = "'전국', '서울', '부산', '대구', '인천', '광주', '대전', '울산', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '세종'"
        # siri.speak()
        return False

    url = 'http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustFrcstDspth'

    today = datetime.today().strftime("%Y-%m-%d")
    params = {'serviceKey': myDriver.userInfo['airKorea_api_key'],
              'returnType': 'json',
              'numOfRows': '10',
              'pageNo': '1',
              'sidoName': target_name,
              'InfomCode': 'PM10',
              'searchDate': today,
              'ver': '1.0'}

    if "SERVICE_KEY_IS_NOT_REGISTERED_ERROR" in requests.get(url, params=params).text:
        myDriver.text = "API key가 잘못되었습니다. 확인바랍니다."
        myDriver.speak()
    else:
        response = json.loads(requests.get(url, params=params).text)
        try:
            cause_text = response['response']['body']['items'][0]['informCause'].split("[미세먼지]")[
                1].strip()
            grade_text = response['response']['body']['items'][0]['informGrade']
            status_dict = change_raw_to_dict(grade_text)
            target_status = status_dict.get(target_name)

            if target_status is not None:
                myDriver.text = f"{target_name}의 미세먼지는 {target_status}입니다."
                myDriver.speak()
                myDriver.text = f'{cause_text}'
                myDriver.speak()
            else:
                myDriver.text = f"찾고자 하는 지역은 {target_name}이지만, 찾을 수 없습니다."
                myDriver.speak()

        except IndexError:
            myDriver.text = "현재 조회된 자료가 없습니다. 나중에 다시 시도해주세요."
            myDriver.speak()

    return True


if __name__ == "__main__":
    print(call_smog("강원", airKorea_api_key='api_key'))
