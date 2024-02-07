# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:01:42 2023

@author: Soo.Y
"""

from src.sql.SQL_handing import *


def doingLogin(myDriver):
    isLogin = True
    while isLogin:
        myDriver.text = "유저 데이터를 불러 옵니다."
        myDriver.speak()
        sql_user_list()
        num_id = input("로그인할 사용자의 id 번호를 눌러주세요. : ")
        if num_id == "new":
            myDriver.text = "신규를 선택하셨습니다."
            myDriver.speak()
            sql_new_user()

        elif num_id == "edit":
            myDriver.text = "수정을 선택하셨습니다. 수정하고자 하는 id를 입력해주세요."
            myDriver.speak()
            num_id = int(input('>>>'))
            _, user_variables = sql_user_gain(num_id)

            sql_update(user_variables)

        elif num_id == "del":
            myDriver.text = "삭제를 선택하셨습니다. 삭제하고자 하는 id를 입력해주세요."
            myDriver.speak()
            num_id = int(input('>>>'))
            _, user_variables = sql_user_gain(num_id)

            sql_delete(user_variables)

        else:
            try:
                isLogin, myDriver.userInfo = sql_user_gain(int(num_id))
            except ValueError:
                myDriver.txt = "숫자를 입력해주세요."
                myDriver.speak()

    return myDriver


if __name__ == "__main__":
    doingLogin(siri)
