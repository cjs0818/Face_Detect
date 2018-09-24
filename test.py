from multiprocessing import Process, Queue
import cv2
import os
import time

from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils


'''
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

def tts_multiproc(msg):

    proc = os.getpid()

    tmpPlayPath = './tmp.mp3'

    # 외부 프로그램 사용 playsound or vlc
    if (os_name.find('Darwin') >= 0):
        playsound(tmpPlayPath)  # For OSX
    else:
        os.system('cvlc ' + tmpPlayPath + ' --play-and-exit')  # For Linux

# ----------------------------------------------------
# 메인 함수
# ----------------------------------------------------
if __name__ == '__main__':

    procs = []
    while True:
        message = "안녕하세요"

        for proc in procs:
            proc.join()
            procs.pop()
        proc = Process(target=tts_multiproc, args=(message,))
        # proc = Process(target=tts.play_proc, args=(message,))
        procs.append(proc)
        proc.start()
        print("Proc started! proc: ", proc, "  len(procs): ", len(procs))

        print(message)

        key_in = cv2.waitKey(20) & 0xFF
        if key_in == ord('q'):
            break
'''

# ----------------------------------------------------------
#  You need to setup PYTHONPATH to include tts/naver_tts.py
#   ex.) export PYTHONPATH=/Users/jschoi/work/ChatBot/Receptionbot_Danbee/receptionbot:$PYTHONPATH
from tts.naver_tts import NaverTTS  # TTS: NaverTTS



def cam_loop(queue_from_cam):
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    while True:
        ret, frame = cap.read()
        #print(ret)
        if(ret):
            #print("len(frame): ", len(frame), ",  type(frame): ", type(frame))
            queue_from_cam.put(frame)
        time.sleep(0.1)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

def main():
    # --------------------------------
    # Create NaverTTS Class
    tts = NaverTTS(0,-1)    # Create a NaverTTS() class from tts/naver_tts.py
    #tts.play("안녕하십니까?")

    procs = []


    #queue_from_cam = Queue()
    #cam_process = Process(target=cam_loop, args=(queue_from_cam, ))
    #cam_process.start()

    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    while(True):
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        #frame = vs.read()
        #rame = imutils.resize(frame, width=640)

        ret, frame = cap.read()

        #ret, frame = cap.read()
        #while queue_from_cam.empty():
        #   pass
        #frame = queue_from_cam.get()
        #print(type(frame))


        message = "안녕하세요"

        for proc in procs:
            proc.join()
            procs.pop()
        proc = Process(target=tts.play, args=(message,))
        procs.append(proc)
        proc.start()
        print("Proc started! proc: ", proc, "  len(procs): ", len(procs))

        print(message)


        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        # update the FPS counter
        fps.update()

    #cam_process.join()

    # When everything done, release the capture
    #cap.release()
    cv2.destroyAllWindows()


    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    #cv2.destroyAllWindows()
    vs.stop()

# ----------------------------------------------------
# 메인 함수
# ----------------------------------------------------
if __name__ == '__main__':
    main()
