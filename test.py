from multiprocessing import Process
import os


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

# ----------------------------------------------------
# 메인 함수
# ----------------------------------------------------
if __name__ == '__main__':
    # --------------------------------
    # Create NaverTTS Class
    tts = NaverTTS(0,-1)    # Create a NaverTTS() class from tts/naver_tts.py
    #tts.play("안녕하십니까?")

    procs = []
    while True:
        message = "안녕하세요"

        for proc in procs:
            proc.join()
            procs.pop()
        proc = Process(target=tts.play, args=(message,))
        procs.append(proc)
        proc.start()
        print("Proc started! proc: ", proc, "  len(procs): ", len(procs))

        print(message)

