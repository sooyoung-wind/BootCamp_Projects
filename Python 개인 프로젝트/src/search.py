# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:08:49 2023

@author: Soo.Y
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchDriverException
from src.common.call_api import call_api
import time


def my_search(target_url="https://www.naver.com", myDriver=None) -> None:
    '''
    Parameters
    ----------
    target_url : str, optional
        DESCRIPTION. The default is "https://www.naver.com".

    Returns
    -------
    None.

    '''
    # myPath_chromedriver = call_api('chromedriver_path')
    myPath_chromedriver = myDriver.userInfo['chromedriver_path']

    try:
        if hasattr(myDriver, 'driver'):
            myDriver.driver.get(target_url)

        else:
            service = Service(executable_path=myPath_chromedriver)
            options = webdriver.ChromeOptions()
            options.add_experimental_option("detach", True)  # 종료 방지
            driver = webdriver.Chrome(service=service, options=options)

            driver.get(target_url)
            return driver
        # time.sleep(10)
    except NoSuchDriverException:
        print("구글 드라이버를 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        print(f"입력된 경로 : {myPath_chromedriver}")


if __name__ == "__main__":
    my_search("https://www.google.com")
