# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:38:32 2023

@author: Soo.Y
"""
import requests
import json
from src.common.call_lat_long import call_lat_long


def call_weather_info(target_name: str, myDriver) -> None:
    celsius_symbol = '\u2103'

    # gps lat, long call method
    try:
        latitude, longitude = call_lat_long(target_name, myDriver.userInfo['googlemaps_api_key'])
        latitude = "latitude=" + str(latitude)
        longitude = "longitude=" + str(longitude)
    except IndexError:
        myDriver.text = '검색이 안되는 주소입니다. 기본으로 설정된 주소가 입력됩니다.'
        myDriver.speak()
        latitude = "latitude=33.3846216"
        longitude = "longitude=126.5534925"

    based_url = 'https://api.open-meteo.com/v1/forecast?'
    my_default = 'daily=temperature_2m_max,temperature_2m_min,rain_sum&windspeed_unit=ms&timezone=Asia%2FTokyo&forecast_days=1'

    total_url = based_url + latitude + "&" + longitude + "&" + my_default

    weather_info = json.loads(requests.get(total_url).text)

    rain = weather_info['daily']['rain_sum'][0]
    temp_max = weather_info['daily']['temperature_2m_max'][0]
    temp_min = weather_info['daily']['temperature_2m_min'][0]

    if rain == 0:
        rain_text = ''
    else:
        rain_text = f'오늘은 비가 옵니다. 강수량은 {rain}mm입니다.'

    temp_max_text = f'오늘 최대 온도는 {temp_max}{celsius_symbol}이고 '
    temp_min_text = f'최저 온도는 {temp_min}{celsius_symbol}입니다.'

    results_text = rain_text + temp_max_text + temp_min_text
    myDriver.text = results_text
    myDriver.speak()


if __name__ == "__main__":
    call_weather_info("제주", googlemaps_api_key='ddd')
