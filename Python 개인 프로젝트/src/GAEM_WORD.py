# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 18:00:58 2023

@author: Soo.Y
"""

import sqlite3
from src.my_speech import MySpeech
import random


class Game_condition:
    def __init__(self):
        self.isWordChecking = True
        self.isDoGaming = True
        self.coin = 3
        self.npcWord = ""
        self.gameCount = 0

    def compare_word(self, word_start):
        return self.npcWord.endswith(word_start[0])

    def end_condition(self, myDriver):
        if self.coin == 0:
            myDriver.text = "당신은 졌습니다. 게임을 종료합니다."
            myDriver.speak()
            self.isWordChecking = False
            self.isDoGaming = False


def gameRun():
    gg = Game_condition()

    npc = MySpeech()
    myDriver = MySpeech()

    ##########################
    # API 접속 구간
    conn = sqlite3.connect("./DB/kr_kp_korean.db")
    cur = conn.cursor()
    kp_words = cur.execute("SELECT word FROM kp WHERE part='명사' ").fetchall()
    kr_words = cur.execute("SELECT word FROM kr WHERE part='명사' ").fetchall()
    conn.close()
    ##########################

    npc.text = "끝말잇기 게임을 시작하겠습니다."
    npc.speak()
    npc.text = "게임의 순서는 플레이어가 먼저 말하고 그 다음 제가 말하겠습니다."
    npc.speak()
    npc.text = "저는 북한말도 알고 있습니다. 다소 어려울 수 있습니다."
    npc.speak()
    npc.text = "당신의 코인 수를 말해주세요."
    npc.speak()
    print(">>>")
    npc.voice_to_text()
    try:
        gg.coin = int(npc.text)
        npc.text = f"당신의 코인 수는 {gg.coin}개입니다."
        npc.speak()
    except ValueError:
        npc.text = "잘못입력된거 같습니다. 임의로 3개의 코인을 넣어두겠습니다."
        npc.speak()
        gg.coin = int(3)

    npc.text = "시작하겠습니다"
    npc.speak()

    # 플레이어

    while gg.isDoGaming:
        # User first
        gg.end_condition(myDriver)
        myDriver.voice_to_text()

        gg.gameCount += 1
        gg.isWordChecking = True

        while gg.isWordChecking:

            if gg.gameCount > 1:
                if not gg.compare_word(myDriver.text):
                    gg.isWordChecking = False
                    npc.text = "제가 말한 마지막 글자와 다릅니다. 코인 1개 감소합니다."
                    gg.coin -= 1
                    npc.speak()
                    gg.end_condition(myDriver)

            if gg.isWordChecking:
                for kp_word in kp_words:
                    if myDriver.text in (kp_word[0]):
                        gg.isWordChecking = False

                for kr_word in kr_words:
                    if myDriver.text in (kr_word[0]):
                        gg.isWordChecking = False

                if gg.isWordChecking:
                    gg.coin -= 1
                    npc.text = f"{myDriver.text} 사전에 존재하는 않는 단어입니다. 코인 1개 감소합니다."
                    npc.speak()
                    gg.isWordChecking = False
                    gg.end_condition(myDriver)

        if gg.isDoGaming:
            word_list = []
            for kp_word in kp_words:
                if myDriver.text[-1] in kp_word[0][0]:
                    word_list.append(kp_word[0])
            for kr_word in kr_words:
                if myDriver.text[-1] in kr_word[0][0]:
                    word_list.append(kr_word[0])

            try:
                gg.npcWord = word_list[random.randint(0, len(word_list))]
                npc.text = gg.npcWord
                npc.speak()
            except:
                npc.text = "당신은 승리했습니다. 게임을 종료합니다."
                npc.speak()
                gg.isDoGaming = False

            if gg.isDoGaming:
                print("="*20)
                npc.text = "플레이어 차례입니다."
                npc.speak()
