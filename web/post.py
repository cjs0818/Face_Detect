# app.py
import requests
import json
import time
import os

# ----------------------------------------------------------
#  You need to setup PYTHONPATH to include tts/naver_tts.py
#   ex.) export PYTHONPATH=/Users/jschoi/work/ChatBot/Receptionbot_Danbee/receptionbot:$PYTHONPATH
from tts.naver_tts import NaverTTS  # TTS: NaverTTS


class Web_API():
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.data_header = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': 'Bearer 9d10041bdb9c4c68a88b7899ca1540c1'  # Dialogflow의 Client access token 입력
        }

    def send_post(self, data_send, url):

        #url = 'http://localhost:60000/message'

        res = requests.post(url,
                            data=json.dumps(data_send),
                            headers=self.data_header)
        return res


def test():
    # --------------------------------
    # Create NaverTTS Class
    tts = NaverTTS(0,-1)    # Create a NaverTTS() class from tts/naver_tts.py
    message = '최종석님, 안녕하세요. 반갑습니다.'
    tts.play(message, False)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


    # --------------------
    #  Body
    video_path = BASE_DIR + "/../animation/ani01_known_approach.mov"
    ani_parameter = {
        'video_path': video_path,
        'pause': 0,
        'audio_enable': 0,
        'video_delay': 90,
        'audio_length': len(message)
    }
    data_send = {
        'speech': message,
        'param': ani_parameter
    }
    # --------------------



    data_header = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': 'Bearer 9d10041bdb9c4c68a88b7899ca1540c1'  # Dialogflow의 Client access token 입력
    }

    url = 'http://localhost:60000/message'


    res = requests.post(url,
                        data=json.dumps(data_send),
                        headers=data_header)

    print(res)

    delay = len(message) * 0.2
    print("delay = %5.3f" % delay)
    time.sleep(delay)



if __name__ == '__main__':

    web_api = Web_API()

    message = '최종석님, 안녕하세요. 반갑습니다.'

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    video_path = BASE_DIR + "/../animation/ani01_known_approach.mov"
    ani_parameter = {
        'video_path': video_path,
        'pause': 0,
        'audio_enable': 0,
        'video_delay': 90,
        'audio_length': len(message)
    }
    data_send = {
        'speech': message,
        'param': ani_parameter
    }
    url = 'http://localhost:60000/message'
    web_api.send_post(data_send, url)


    # --------------------------------
    # Create NaverTTS Class
    tts = NaverTTS(0,-1)    # Create a NaverTTS() class from tts/naver_tts.py
    tts.play(message)



    #test()
