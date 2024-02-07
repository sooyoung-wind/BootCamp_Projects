# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:41:04 2023

@author: Soo.Y
"""

import os
from dotenv import load_dotenv


def call_api(api_name: str, env_path='../../env/data.env') -> str:
    """
    Parameters
    ----------
    api_name : str
        환경변수 이름.
    env_path : str, 
        환경변수 파일경로. The default is 'env/data.env'.

    Returns
    -------
    Key 값 리턴

    """
    if os.path.isfile(env_path):
        print('api key를 로드합니다.')
        load_dotenv(env_path)
        if os.getenv(api_name):
            print(f'성공적으로 {api_name}를 로드했습니다.')
            return os.getenv(api_name)
        else:
            print(f'요청하신 {api_name}가 존재하지 않습니다.')
            return input('api key를 직접 입력해주세요 :')
    else:
        print(f'요청하신 {api_name}가 존재하지 않습니다.')
        return input('api key를 직접 입력해주세요 :')


if __name__ == "__main__":
    call_api('googlemaps_api_key', env_path='../../../env/data.env')
