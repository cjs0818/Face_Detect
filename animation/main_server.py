# -*- coding: utf-8 -*-

# ----------------------------
# To Play Video: The first frame
#from animation.play_animation import Play_AV

from flask import Flask, request, jsonify, Response
import cv2

from multiprocessing import Process, Queue
import os



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ani_parameter = {
    'video_path': BASE_DIR + "/ani01_known_approach.mov",
    'pause': 1,
    'audio_enable': 0,
    'video_delay': 90,
    'audio_length': 0
}
q_param = Queue()
q_param.put(ani_parameter)

class Play_AV():
    def __init__(self, video_path, winname, x, y, video_delay=90):
        self.param = dict()
        self.idle_video_path = video_path
        self.winname = winname
        self.win_pos_x = x
        self.win_pos_y = y
        self.video_delay = video_delay

        self.video = cv2.VideoCapture(video_path)
        grabbed, play_frame = self.video.read()

        cv2.namedWindow(winname)
        cv2.moveWindow(winname, x, y)


        # ------------------------------
        #   Skip the first frame
        grabbed, play_frame = self.video.read()
        cv2.imshow(winname, play_frame)
        # ------------------------------

    def play_av_idle(self):
        winname = self.winname

        try:
            video = self.video
            grabbed, play_frame = video.read()
            cv2.imshow(winname, play_frame)
        except:     # When reaaching the end of the idle_video_file
            self.video = cv2.VideoCapture(self.idle_video_path)
            video = self.video
            grabbed, play_frame = video.read()
            grabbed, play_frame = video.read()
            cv2.imshow(winname, play_frame)
            #print("Read Idle Video File Expired!")
            pass


        key_in = cv2.waitKey(self.video_delay) & 0xFF
        return key_in


    def play_av_normal(self, video_path, pause=0, audio_enable=0, video_delay=3, audio_length=0):

        self.video.release
        self.video = cv2.VideoCapture(video_path)
        video = self.video
        winname = self.winname

        # ---------
        if (pause == 1):
            winname = self.winname
            grabbed, play_frame = video.read()
            cv2.imshow(winname, play_frame)
        else:
            if(audio_enable == 1):
                player = MediaPlayer(video_path)

            cnt = 0
            cnt_th = int(audio_length / 15) + 1

            while cnt < cnt_th:
                cnt += 1

                while True:
                    grabbed, play_frame = video.read()
                    if (audio_enable == 1):
                        audio_frame, val = player.get_frame()
                    if not grabbed:
                        print("End of video")
                        break
                    key_in = cv2.waitKey(video_delay) & 0xFF
                    if key_in == ord("q"):
                        return key_in
                    cv2.imshow(winname, play_frame)
                    if (audio_enable == 1):
                        if val != 'eof' and audio_frame is not None:
                            # audio
                            img, t = audio_frame

        # ---------
        self.video.release
        self.video = cv2.VideoCapture(self.idle_video_path)

        return 0


def animation(q):   # multiprocessing Process with Queue
    video_path = BASE_DIR + "/ani01_idle.mov"
    winname = "Video"
    win_pos_x = 0
    win_pos_y = 10
    video_delay = 90
    av = Play_AV(video_path, winname, win_pos_x, win_pos_y, video_delay)

    while True:

        if(q.empty()):
            key_in = av.play_av_idle()
        else:
            param = q.get()

            video_path = param['video_path']
            pause = param['pause']
            audio_enable = param['audio_enable']
            video_delay = param['video_delay']
            audio_length = param['audio_length']

            key_in = av.play_av_normal(video_path, pause, audio_enable, video_delay, audio_length)

        if key_in == ord("q"):
            break

    #cv2.destroyAllWindows()


app = Flask('__name__')

#@app.route('/message', methods=['POST', 'GET'])
@app.route('/message', methods=['POST'])
def message():

    res = request.get_json(force=True)
    answer = res['speech']
    print (answer)

    #return jsonify(res)
    #return answer

    ani_parameter = {
        'video_path': BASE_DIR + "/ani01_known_approach.mov",
        'pause': 0,
        'audio_enable': 0,
        'video_delay': 90,
        'audio_length': len("최종석님, 안녕하십니까? 반갑습니다.")
    }

    q_param.put(ani_parameter)


    resp = Response('로그인 성공!',status=200)
    #print (resp)
    return resp




if __name__ == '__main__':

    proc = Process(target=animation, args=(q_param, ))
    proc.start()
    print("  --------    Proc started")


    app.run(host='0.0.0.0', port=60000, debug=True)

    proc.join()
