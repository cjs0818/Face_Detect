# -*- coding: utf-8 -*-
# for amazon
import re
import os
import sys
import time
from boto3 import client    # pip3 install boto3
from botocore.exceptions import BotoCoreError, ClientError
#import vlc
from playsound import playsound  #  pip3 install pyobjc, playsound
from pyssml.PySSML import PySSML

languageModel = ["Seoyeon", "Joanna"]

KOREAN = 0
ENGLISH = 1

# 한글 구분
def isHangul(text):
    #Check the Python Version
    pyVer3 =  sys.version_info >= (3, 0)

    if pyVer3 : # for Ver 3 or later
        encText = text
    else: # for Ver 2.x
        if type(text) is not unicode:
            encText = text.decode('utf-8')
        else:
            encText = text

    hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', encText))
    return hanCount > 0


# 아마존 서비스 코드
# isSSML을 True로 보내주면
# ssml 방식으로
# False인 경우 일반 Text 방식으로 요청하게 된다.
def aws_polly(text, isSSML = False):
    # check Hangul
    if isHangul(text):
        # Korean
        voiceid = languageModel[KOREAN]
    else:
        # English
        voiceid = languageModel[ENGLISH]

    try:
        polly = client("polly", region_name="ap-northeast-2")
        '''
        response = client.synthesize_speech(
            LexiconNames=[
                'string',
            ],
            OutputFormat='json'|'mp3'|'ogg_vorbis'|'pcm',
            SampleRate='string',
            SpeechMarkTypes=[
                'sentence'|'ssml'|'viseme'|'word',
            ],
            Text='string',
            TextType='ssml'|'text',
            VoiceId='Geraint'|'Gwyneth'|'Mads'|'Naja'|'Hans'|'Marlene'|'Nicole'|'Russell'|'Amy'|'Brian'|'Emma'|'Raveena'|'Ivy'|'Joanna'|'Joey'|'Justin'|'Kendra'|'Kimberly'|'Matthew'|'Salli'|'Conchita'|'Enrique'|'Miguel'|'Penelope'|'Chantal'|'Celine'|'Mathieu'|'Dora'|'Karl'|'Carla'|'Giorgio'|'Mizuki'|'Liv'|'Lotte'|'Ruben'|'Ewa'|'Jacek'|'Jan'|'Maja'|'Ricardo'|'Vitoria'|'Cristiano'|'Ines'|'Carmen'|'Maxim'|'Tatyana'|'Astrid'|'Filiz'|'Vicki'|'Takumi'|'Seoyeon'|'Aditi'
        )
        '''
        if isSSML:
            textType = 'ssml'
        else:
            textType = 'text'

        response = polly.synthesize_speech(
                TextType=textType,
                Text=text,
                OutputFormat="mp3",
                VoiceId=voiceid)

        # get Audio Stream (mp3 format)
        stream = response.get("AudioStream")

        # save the audio Stream File
        with open('aws_test_tts.mp3', 'wb') as f:
            data = stream.read()
            f.write(data)


        # VLC 플레이어를 이용해서 오디오 출력
        # non block
        #p = vlc.MediaPlayer('./aws_test_tts.mp3')
        #p.play()

        playsound('./aws_test_tts.mp3')  # OSX: pip3 install pyobjc, playsound

    except ( BotoCoreError, ClientError) as err:
        print(str(err))


# 호출 코드
if __name__ == '__main__':
    # pyssml 생성
    # s = PySSML()
    # 아마존 speech ssml
    s = AmazonSpeech()
    # 일반
    s.say('오늘의 ')

    # <prosody> 태그 속도를 최대한 느리게.
    s.prosody({'rate': "x-slow"}, '날씨를 전해 드리겠습니다.')

    s.say('현재, 전국이 구름이 많은 가운데 일부 중부 지역과 전북에는 ')

    # <prosody> 태그 음성을 최대한 크게.
    s.prosody({'volume': 'x-loud'}, '눈이 날리거나')

    # <pause> 태그 1초 쉼
    s.pause('1s')

    # <prosody> 태그 음 높이? 음색을 최대한 높게
    s.prosody({'pitch': 'x-high'}, '빗방울이 떨어지는 곳이 있습니다.')

    s.say(' 서울의 경우 북부 지역을 중심으로')

    # <prosody> 태그 속도를 10% 빠르게.
    s.prosody({'rate': '10%'}, '눈이 날리고 있으나,')

    s.say('공식적인 첫눈으로 기록되지는 않습니다.')

    # 아마존 효과 태그 적용 속삭임
    s.whisper('속삭이는 중입니다.')
    # ssml 방식으로 문자열 출력
    print(s.ssml())
    aws_polly(s.ssml(), True)

    # 음성 출력하는 동안 대기
    time.sleep(50)
