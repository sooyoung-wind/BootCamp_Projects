# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:21:08 2023

@author: Soo.Y
"""


from gtts import gTTS
import playsound
import os
import speech_recognition as sr


class MySpeech:
    def __init__(self, text="안녕하세요. 부팅이 완료되었습니다."):
        self.text = text
        pass

    def speak(self):
        tts = gTTS(text=self.text, lang='ko')
        tts.save('voice.mp3')
        print(self.text)
        playsound.playsound('voice.mp3')
        os.remove('voice.mp3')

    def voice_to_text(self):
        r = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            print("마이크 인식 중...")
            # audio = r.listen(source, timeout=5, phrase_time_limit=5)
            audio = r.listen(source)

        try:
            self.text = r.recognize_google(audio, language='ko-KR')
            print('입력 : ' + self.text)
        except sr.UnknownValueError:
            print('음성 인식 실패')
        except sr.RequestError:
            print("서버 에러 발생")
        except sr.WaitTimeoutError:
            print("인식 실패")


if __name__ == "__main__":
    ab = MySpeech()
    ab.speak("안녕하세요.")
    ab.voice_to_text()
