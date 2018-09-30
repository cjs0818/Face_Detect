# -*- coding: utf-8 -*-

# 네이버 음성합성 Open API 예제
import os
import urllib.request

#------------------
# Sound player
#
#    Linux  -> use cvlc :
#       sudo apt-get install vlc
#    OSX -> use playsound :
#       pip3 install pyobjc
#       pip3 install playsound
#
#
import subprocess
os_name = subprocess.check_output('uname', shell=True)
os_name = str(os_name)
if(os_name.find('Darwin') >= 0):
    from playsound import playsound  # For OSX
#------------------


#----------------------------------------------------------
#   다음의 변수를 세팅해야 함:
#    doc폴더에 있는 google_naver_key.pdf (암호 필요)를 열면 아래 정보가 나와 있음
#
# client_id = "xxxxxxxxxxx"
# client_secret = "yyyyyyyyyyyyyy"
#----------------------------------------------------------

# ---------------------------------------------------------
#    Naver Cloud Platform (https://console.ncloud.com)
#       최종석 개인용
client_id = "eyrxb9rg98"
client_secret = "DK9FvMgRTGBYu1IYLYMpUCniGoku8iaQ5e7bHi1D"
# ---------------------------------------------------------

# ---------------------------------------------------------
#    Naver Cloud Platform (https://console.ncloud.com)
#       sHRI Lab 공용
client_id = "wdkz6cincl"
client_secret = "LZ3N99fb48HsAKoIQ6jExrcyvBiV1fXYOc9Fbgrk"
# ---------------------------------------------------------


url = "https://naveropenapi.apigw.ntruss.com/voice/v1/tts"

speakers = [
    'mijin',     #한국어 여성
    'jinho',     #한국어 남성
    'clara',     #영어 여성
    'matt',      #영어 남성
    'yuri',      #일본어 여성
    'shinji',    #일본어 남성
    'meimei',    #중국어 여성
    'liangliang',#중국어 남성
    'jose',      #스페인어 남성
    'carmen'     #스페인어 여성
    ]


tmpPlayPath = './tmp.mp3'

class NaverTTS():
    def __init__(self, speaker=0, speed=0):
        self.speaker = speakers[speaker]
        self.speed=str(speed)
    def play(self, txt):
        encText = urllib.parse.quote(txt)
        data = "speaker=" + self.speaker + "&speed=" + self.speed + "&text=" + encText;

        request = urllib.request.Request(url)
        request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
        request.add_header("X-NCP-APIGW-API-KEY",client_secret)
        response = urllib.request.urlopen(request, data=data.encode('utf-8'))
        rescode = response.getcode()
        if(rescode==200):
            response_body = response.read()
            with open(tmpPlayPath, 'wb') as f:
                f.write(response_body)

            #외부 프로그램 사용 playsound or vlc
            if(os_name.find('Darwin') >= 0):
                playsound(tmpPlayPath)  # For OSX
            else:
                os.system('cvlc ' + tmpPlayPath + ' --play-and-exit') # For Linux


            #라즈베리파이
            #os.system('omxplayer ' + tmpPlayPath)
        else:
            print("Error Code:" + rescode)



def main():
    tts = NaverTTS()
    tts.play("안녕하십니까?")

if __name__ == '__main__':
    main()
