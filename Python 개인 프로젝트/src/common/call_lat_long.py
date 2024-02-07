# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:45:45 2023

@author: Soo.Y
"""
import googlemaps
from typing import List


def call_lat_long(adress: str, googlemaps_api_key: str) -> List[float]:
    '''
    Parameters
    ----------
    adress : str
        주소 예) 삼도이동 1260 .
    googlemaps_api_key : str
        DESCRIPTION.

    Returns
    -------
    List[float]
        좌표 값
    '''
    try:
        gmaps = googlemaps.Client(key=googlemaps_api_key)
        geocode_results = gmaps.geocode(adress, language='ko')

        my_lat = geocode_results[0]['geometry']['location']['lat']
        my_long = geocode_results[0]['geometry']['location']['lng']
        return [my_lat, my_long]
    except ValueError:
        print("API key가 잘못되었습니다. 확인바랍니다.")
        return False


if __name__ == "__main__":
    print(call_lat_long("제주", googlemaps_api_key='api_key'))
