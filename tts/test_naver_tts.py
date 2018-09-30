# -*- coding: utf-8 -*-

# 네이버 음성합성 Open API 예제
from naver_tts import NaverTTS   # <- from tts.naver_tts import NaverTTS

def main():
    tts = NaverTTS()
    tts.play("안녕하십니까?")

if __name__ == '__main__':
    main()
