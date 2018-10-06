# -*- coding: utf-8 -*-

from ffpyplayer.player import MediaPlayer
import cv2

# ----------------------------------------------------------
#  You need to setup PYTHONPATH to include tts/naver_tts.py
#   ex.) export PYTHONPATH=/Users/jschoi/work/ChatBot/Receptionbot_Danbee/receptionbot:$PYTHONPATH
from tts.naver_tts import NaverTTS  # TTS: NaverTTS


class Play_AV():
    def __init__(self):
        self.audio_enable = 0


    '''
    #---------- Error!!! ----------
    #  File "/Users/jschoi/work/ChatBot/Face_Detect/animation/play_animation.py", line 41, in play_av
    #   cv2.imshow(winname, play_frame)
    #   cv2.error: OpenCV(3.4.3) /Users/travis/build/skvark/opencv-python/opencv/modules/highgui/src/window.cpp:356: error: (-215:Assertion failed) size.width>0 && size.height>0 in function 'imshow'
    #------------------------------
    def play_av(self, video_path, pause=0, audio_enable=0, video_delay=3, audio_length=0):
        cnt = 0
        cnt_th = int(audio_length / 15) + 1

        # video_path = './animation/csy02.mov'
        video = cv2.VideoCapture(video_path)
        grabbed, play_frame = video.read()
        winname = "Video"
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 0, 10)
        cv2.imshow(winname, play_frame)


        while cnt < cnt_th:
            cnt += 1

            #video_path = './animation/csy02.mov'
            #video = cv2.VideoCapture(video_path)
            if(audio_enable == 1):
                player = MediaPlayer(video_path)

            grabbed, play_frame = video.read()
            winname = "Video"
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 0, 10)
            cv2.imshow(winname, play_frame)


            if pause == 0:
                while True:
                    grabbed, play_frame = video.read()
                    if (audio_enable == 1):
                        audio_frame, val = player.get_frame()
                    if not grabbed:
                        print("End of video")
                        break
                    if cv2.waitKey(video_delay) & 0xFF == ord("q"):
                        break
                    cv2.imshow("Video", play_frame)
                    if (audio_enable == 1):
                        if val != 'eof' and audio_frame is not None:
                            # audio
                            img, t = audio_frame
        video.release()
    '''


    def play_av(self, video_path, pause=0, audio_enable=0, video_delay=3, audio_length=0):
        cnt = 0
        cnt_th = int(audio_length / 15) + 1

        while cnt < cnt_th:
            cnt += 1

            #video_path = './animation/csy02.mov'
            video = cv2.VideoCapture(video_path)
            if(audio_enable == 1):
                player = MediaPlayer(video_path)

            grabbed, play_frame = video.read()
            winname = "Video"
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 0, 10)
            cv2.imshow(winname, play_frame)


            if pause == 0:
                while True:
                    grabbed, play_frame = video.read()
                    if (audio_enable == 1):
                        audio_frame, val = player.get_frame()
                    if not grabbed:
                        print("End of video")
                        break
                    if cv2.waitKey(video_delay) & 0xFF == ord("q"):
                        break
                    cv2.imshow("Video", play_frame)
                    if (audio_enable == 1):
                        if val != 'eof' and audio_frame is not None:
                            # audio
                            img, t = audio_frame
        video.release()


#----------------------------------------------------
# 메인 함수
#----------------------------------------------------
if __name__ == '__main__':

    av = Play_AV()

    video_path = './csy02_Hi_Short.mp4'
    audio_enable = 0
    pause = 1
    av.play_av(video_path, pause, audio_enable)


    # ===============================
    # Create NaverTTS Class
    tts = NaverTTS(0,-1)    # Create a NaverTTS() class from tts/naver_tts.py

    message = "안녕하세요. 반갑습니다. 두번째로 반가워요. 세번째로 반갑습니다."
    tts.play(message, False)

    # ----------------------------
    # To Play Video
    video_path = './csy02_Hi_Short.mp4'
    audio_enable = 0
    pause = 0
    video_delay = 60
    print(len(message))
    av.play_av(video_path, pause, audio_enable, video_delay, len(message))
    # ----------------------------
    # -------------------------------

