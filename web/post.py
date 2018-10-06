# app.py
import requests
import json
import time

# ----------------------------------------------------------
#  You need to setup PYTHONPATH to include tts/naver_tts.py
#   ex.) export PYTHONPATH=/Users/jschoi/work/ChatBot/Receptionbot_Danbee/receptionbot:$PYTHONPATH
from tts.naver_tts import NaverTTS  # TTS: NaverTTS


def main():
    # --------------------------------
    # Create NaverTTS Class
    tts = NaverTTS(0,-1)    # Create a NaverTTS() class from tts/naver_tts.py
    message = '최종석님, 안녕하세요. 반갑습니다.'
    tts.play(message, False)

    ani_parameter = {
        'video_path': './animation/ani01_known_approach.mov',
        'pause': 0,
        'audio_enable': 0,
        'video_delay': 90,
        'audio_length': len("최종석님, 안녕하십니까? 반갑습니다.")
    }

    data_send = {
        'speech': message,
        'param': ani_parameter,
    }

    data_header = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': 'Bearer 9d10041bdb9c4c68a88b7899ca1540c1'  # Dialogflow의 Client access token 입력
    }

    url = 'http://localhost:60000/message'


    res = requests.post(url,
                        data=json.dumps(data_send),
                        headers=data_header)

    delay = len(message) * 0.2
    print(delay)
    time.sleep(delay)

if __name__ == '__main__':
    main()
