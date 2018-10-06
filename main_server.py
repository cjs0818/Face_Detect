# ----------------------------
# To Play Video: The first frame
from animation.play_animation import Play_AV

from flask import Flask, request, jsonify, Response, render_template
import vlc
import cv2

from multiprocessing import Process, Pool, Queue
import time


app = Flask('__name__')


q = Queue()

ani_parameter = {
    'video_path': './animation/ani01_known_approach.mov',
    'pause': 1,
    'audio_enable': 0,
    'video_delay': 90
}

q.put(ani_parameter)

def animation(q):

    '''
    for i in range(50000):
        print(i)
        time.sleep(1)
    '''

    av = Play_AV()

    while True:

        if(q.empty()):
            print("Empty!")

        param = q.get()

        video_path = param['video_path']
        pause = param['pause']
        audio_enable = param['audio_enable']
        video_delay = param['video_delay']

        # ----------------------------
        # To Play Video: The first frame
        #av = Play_AV()
        #video_path = './animation/ani01_known_approach.mov'
        #pause = 0
        #audio_enable = 1
        #video_delay = 100
        av.play_av(video_path, pause, audio_enable, video_delay)
        # ----------------------------


    #cv2.destroyAllWindows()



#@app.route('/message', methods=['POST', 'GET'])
@app.route('/message', methods=['POST'])
def message():

    res = request.get_json(force=True)
    answer = res['query']
    print (answer)

    #return jsonify(res)
    #return answer

    ani_parameter = {
        'video_path': './animation/ani01_known_approach.mov',
        'pause': 0,
        'audio_enable': 0,
        'video_delay': 90
    }

    q.put(ani_parameter)


    resp = Response('로그인 성공!',status=200)
    #print (resp)
    return resp




if __name__ == '__main__':


    proc = Process(target=animation, args=(q, ))
    proc.start()
    print("  --------    Proc started")

    app.run(host='0.0.0.0', port=60000, debug=True)

    proc.join()
