# -*- coding: utf-8 -*-

from ffpyplayer.player import MediaPlayer
import cv2

class Play_AV():
    def __init__(self):
        self.audio_enable = 0

    def play_av(self, video_path, pause=0, audio_enable=0):
        #video_path = './animation/csy02.mov'
        video = cv2.VideoCapture(video_path)
        if(audio_enable == 1):
            player = MediaPlayer(video_path)

        grabbed, play_frame = video.read()
        cv2.imshow("Video", play_frame)

        if pause == 0:
            while True:
                grabbed, play_frame = video.read()
                if (audio_enable == 1):
                    audio_frame, val = player.get_frame()
                if not grabbed:
                    print("End of video")
                    break
                if cv2.waitKey(3) & 0xFF == ord("q"):
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

    video_path = './csy02.mov'
    audio_enable = 0
    pause = 1
    av.play_av(video_path, pause, audio_enable)

    audio_enable = 1
    pause = 0
    av.play_av(video_path, pause, audio_enable)

