# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 00:36:27 2023

@author: Soo.Y
"""
from konlpy.tag import Okt


def extracting_search_words(text):
    words = Okt().nouns(text)

    exclueded_words = ['네이버', "naver", "검색", "google", "Google", "구글", "미세먼지", "날씨", "지역"]
    filtered_words = [word for word in words if word not in exclueded_words]

    return " ".join(filtered_words)


if __name__ == "__main__":
    print(extracting_search_words("제주시 미세먼지 알려줘"))
