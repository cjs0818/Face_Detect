# -*- coding: utf-8 -*-

# app.py
from flask import Flask, request, jsonify, Response, render_template
import vlc

import cv2
#from camera import VideoCamera



'''
# Rasberry PI
# pip3 install omxplayer-wrapper
#    <- pip3 install dbus-python (in OSX), sudo apt-get install python-dubs (in Linux)
#    <- brew install glib (in OSX)
from omxplayer import OMXPlayer
'''


app = Flask('__name__')


def gen():

    #vlc_instance = vlc.Instance('--no-audio', '--fullscreen')
    vlc_instance = vlc.Instance()
    player = vlc_instance.media_player_new()
    media = vlc_instance.media_new('../animation/ani01_known_approach.mov')
    player.set_media(media)
    player.play()


    '''
    # This will start an `omxplayer` process, this might
    # fail the first time you run it, currently in the
    # process of fixing this though.
    player = OMXPlayer('../animation/ani01_known_approach.mov')


    # The player will initially be paused

    player.play()
    sleep(5)
    player.pause()

    # Kill the `omxplayer` process gracefully.
    player.quit()
    '''


    print("gen() started!")



    '''
    cap = cv2.VideoCapture('./animation/csy02.mov')
    cv2.namedWindow('Video Play', cv2.WINDOW_AUTOSIZE)

    frame_count = 0

    while (cap.isOpened()):
        ret, frame = cap.read()

        frame_count += 1
        print(frame_count)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('frame',gray)
        try:
            cv2.imshow('frame', frame)
        except:
            pass
        # if cv2.waitKey(15) & 0xFF == ord('q'):
        #    break

    cap.release()
    cv2.destroyAllWindows()
    '''

#@app.route('/message', methods=['POST', 'GET'])
@app.route('/message', methods=['POST'])
def message():

    res = request.get_json(force=True)
    answer = res['query']
    print (answer)

    #return jsonify(res)
    #return answer

    resp = Response('로그인 성공!',status=200)
    #print (resp)
    #return resp


    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')








if __name__ == '__main__':
    app.run(host='0.0.0.0', port=60000, debug=True)
