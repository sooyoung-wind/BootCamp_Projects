# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 09:53:05 2023

@author: Soo.Y
"""
import sqlite3


def sql_new_user():
    name = input("사용자 이름을 입력해주세요. :")
    googlemaps_api_key = input("구글맵 API KEY를 입력해주세요. :")
    chromedriver_path = input("크롬드라이버 경로를 입력해주세요. :")
    airKorea_api_key = input("미세먼지 API KEY를 입력해주세요. :")
    call = input("저의 이름을 입력해주세요. :")

    user_variables = {'name': name,
                      'googlemaps_api_key': googlemaps_api_key,
                      'chromedriver_path': chromedriver_path,
                      'airKorea_api_key': airKorea_api_key,
                      'call': call
                      }
    sql_insert(user_variables)


def sql_insert(user_variables):
    conn = sqlite3.connect("DB/siri_userData.db")
    cur = conn.cursor()

    INSERT_SQL = """
        INSERT INTO items(name, googlemaps_api_key,
                         chromedriver_path, airKorea_api_key, call)
          VALUES (?,?,?,?,?)
    """

    insert_data = (user_variables)
    cur.execute(INSERT_SQL, tuple(insert_data.values()))
    conn.commit()
    conn.close()


def sql_delete(user_variables):
    conn = sqlite3.connect("DB/siri_userData.db")
    cur = conn.cursor()

    cur.execute("DELETE FROM items WHERE name=:name",
                {'name': user_variables['name']})
    conn.commit()
    conn.close()


def _making_sentence(update_num):
    if update_num == 2:
        print("사용자 이름을 변경합니다.")
        return 'UPDATE items SET name=? WHERE id=?'
    if update_num == 3:
        print("구글맵 API KEY를 변경합니다.")
        return 'UPDATE items SET googlemaps_api_key=? WHERE id=?'
    if update_num == 4:
        print("크롬드라이버 경로를 변경합니다.")
        return 'UPDATE items SET chromedriver_path=? WHERE id=?'
    if update_num == 5:
        print("미세먼지 API KEY를 변경합니다.")
        return 'UPDATE items SET airKorea_api_key=? WHERE id=?'
    if update_num == 6:
        print("호출어를 변경합니다.")
        return 'UPDATE items SET call=? WHERE id=?'
    if update_num == 7:
        print("메모 내용을 수정합니다.")
        return 'UPDATE items SET note=? WHERE id=?'


def sql_update(user_variables):
    conn = sqlite3.connect("DB/siri_userData.db")
    cur = conn.cursor()

    target_id = sql_retrieval(user_variables)

    isUpdate = True
    while isUpdate:
        try:
            update_num = int(input("수정하고자 하는 번호를 입력해주세요. :"))
            if 2 <= update_num <= 7:
                UPDATE_SQL = _making_sentence(update_num)
                update_value = input('수정하고자 하는 내용을 입력해주세요. :')
                cur.execute(UPDATE_SQL, (update_value, target_id))
                conn.commit()
                isUpdate = False
            else:
                print("잘못입력하셨습니다.")

        except ValueError:
            print("잘못입력하셨습니다.")


def sql_retrieval(user_variables):
    conn = sqlite3.connect("DB/siri_userData.db")
    cur = conn.cursor()

    user_info = cur.execute("SELECT * FROM items WHERE name=:name;",
                            {'name': user_variables['name']}).fetchall()

    for info in user_info:
        sql_id, name, googlemaps_api_key, chromedriver_path, airKorea_api_key, call, note = info

        print(f'1. id: {sql_id}(수정불가능)')
        print(f'2. 사용자: {name}')
        print(f'3. 구글맵 API KEY: {googlemaps_api_key}')
        print(f'4. 크롬드라이버 경로: {chromedriver_path}')
        print(f'5. 미세먼지 API KEY: {airKorea_api_key}')
        print(f'6. 호출어 : {call}')
        print(f'7. note : {note}')
        print('=' * 20)
        temp_id = sql_id
    return temp_id


def sql_user_list():
    import os
    print(os.getcwd())
    conn = sqlite3.connect("DB/siri_userData.db")
    cur = conn.cursor()
    user_info = cur.execute(
        "SELECT id, name FROM items ORDER BY id").fetchall()

    print('-' * 20)
    for info in user_info:
        sql_id, name = info
        print(f'id: {sql_id}  사용자 : {name}')

    print("신규 : new, 수정 : edit, 삭제 : del 를 눌러주세요.")
    print('-' * 20)
    conn.close()


def sql_user_gain(sql_id):
    conn = sqlite3.connect("DB/siri_userData.db")
    cur = conn.cursor()
    user_info = cur.execute(
        "SELECT * FROM items WHERE id=:id;", {'id': sql_id}).fetchall()

    if len(user_info) == 0:
        print("해당 ID가 존재하지 않습니다.")
        return True, None
    else:
        sql_id, name, googlemaps_api_key, chromedriver_path, airKorea_api_key, call, note = user_info[
            0]
        user_variables = {'sql_id': sql_id,
                          'name': name,
                          'googlemaps_api_key': googlemaps_api_key,
                          'chromedriver_path': chromedriver_path,
                          'airKorea_api_key': airKorea_api_key,
                          'call': call,
                          'note': note
                          }
        return False, user_variables


def sql_create():
    print("DB를 새로 생성합니다.")
    conn = sqlite3.connect("DB/siri_userData.db")
    cur = conn.cursor()
    CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS items(
            id integer primary key autoincrement,
            name text not null,
            googlemaps_api_key text not null,
            chromedriver_path text not null,
            airKorea_api_key text not null,
            call text not null,
            note
            )
    """
    cur.execute(CREATE_SQL)
    conn.close()
