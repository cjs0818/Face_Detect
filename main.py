# -*- coding: utf-8 -*-

# ----------------------------------------------
# Face Detection, Face Recognition, and Head Pose Detection: by Dlib
# Camera Capture: by OpenCV
#
#   Face Detection: face_detection/fd_dlib.py
#   Face Recognition: face_recognition/fr_dlib.py
#   Head Pose Detection: hd_detection.py
#   Action Event Detection: action_detection/action_detection.py
# ----------------------------------------------

import numpy as np
import cv2
import pickle    # used for Face recognition by OpenCV
import time


import dlib
import os
from multiprocessing import Process, Queue
import sys



import datetime



#-------------------------------------------------------------
# chatbot/dialogflow.py  for Dialogflow chatbot platform
from chatbot.dialogflow import ChatBot   # Chatbot platform: Dialogflow.ai
from stt.gspeech import Gspeech     # STT: Google Cloud Speech
import json

# ----------------------------------------------------------
#  You need to setup PYTHONPATH to include tts/naver_tts.py
#   ex.) export PYTHONPATH=/Users/jschoi/work/ChatBot/Receptionbot_Danbee/receptionbot:$PYTHONPATH
from tts.naver_tts import NaverTTS  # TTS: NaverTTS


# ----------------------------
# Face detection: by Dlib:
#    face_detection/fd_dlib.py
#detector = dlib.get_frontal_face_detector()

# ----------------------------
# Face Recognition: by Dlib:
#    face_recognition/fr_dlib.py
#  ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
#    You can download a trained facial shape predictor and recognition model from:\n"
#     http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
#     http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
from face_recognition.fr_dlib import FaceRecog

# ----------------------------
# Head Pose detection: by Dlib:
#   hp_detection.py
#   You can download a trained facial shape predictor from:
#    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
from hp_detection import HeadPose


# ----------------------------
#   Action Event Detection: action_detection/action_detection.py
from action_detection.action_detect import Event_Detector
ACTION_STATE_IDLE = 0
ACTION_EVENT_APPROACH = 1
ACTION_EVENT_DISAPPEAR = 2
ACTION_STATE_FACE_DETECTED = 3


'''
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
'''


# ------------------------
# Object Tracking by Dlib correlation_tracker
from tracker.obj_tracker import Obj_Tracker

# ----------------------------
# To Play Video: The first frame
from animation.play_animation import Play_AV

# -----------------------
# Web_API class for web POST
from web.post import Web_API

# -----------------------
# Database
from data.database import Database, MongoDB



'''
#------------------------------
# For Multiprocessing
queue_from_cam = Queue()

def cam_loop(queue_from_cam):
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    while True:
        ret, frame = cap.read()
        queue_from_cam.put(frame)
#------------------------------
'''


def tts_animation(message, tts, av, web_api, gsp, obj_track, param, loop_path=[]):
    tts_enable = param['tts_enable']
    stt_enable = param['stt_enable']
    ani_multiprocessing = param['ani_multiprocessing']
    ad_event = param['ad_event']
    video_path = param['video_path']
    pause = param['pause']
    audio_enable = param['audio_enable']
    video_delay = param['video_delay']
    audio_length = param['audio_length']

    # ===============================
    # TTS
    if tts_enable == 1:
        obj_track.track_started = False
        if stt_enable == 1:  # TTS 하는 동안 STT 일시 중지 --
            gsp.pauseMic()

        # ----------------------------
        # To Play Video

        if ani_multiprocessing == 0:
            block = False
            tts.play(message, block)

            av.play_av(video_path, pause, audio_enable, video_delay)
            # ----------------------------
        else:
            ani_parameter = {
                'video_path': video_path,
                'pause': pause,
                'audio_enable': audio_enable,
                'video_delay': video_delay,
                'audio_length': audio_length
            }
            data_send = {
                'speech': message,
                'param': ani_parameter,
            }
            url = 'http://localhost:60000/message'

            try:
                cnt = 0
                if video_path == loop_path:
                    cnt_th = int(len(message) / 15) + 1
                else:
                    cnt_th = 1

                while cnt < cnt_th:
                    cnt += 1
                    web_api.send_post(data_send, url)
            except:
                print("You must execute main_server.py in 'animation' folder!!! ")
                print("Type Ctrl-c to exit!     SDA")
                input()

            block = True
            tts.play(message, block)


def main(stt_enable=1, tts_enable=1, ani_multiprocessing=1):
    if stt_enable == 1:
        dialog_flag = True  # Enable speech recognition when APPROACH, Disable when dialog ends
        gsp = Gspeech()
    else:
        # 음성인식 아닌 경우, 테스트 query에 대해 문장 단위로 테스트
        query = [
            "사람",
                "아나스타샤를 찾으러 왔어요",
#            "안녕, 안내를 부탁해요",
#            "사람",
#                "최종석 박사님을 만나러 왔어요",
#            "안녕, 안내를 부탁해요",
#            "사람",
#                "홍길동님을 찾으러 왔어요",
#            "안녕, 안내를 부탁해요",
#            "사람",
#                "여진구 박사님이요",
                "끝내자"
                 ]
        q_length = len(query)
        q_iter = 0
        dialog_flag = q_iter < q_length


    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    ret, sample_frame = cap.read()

    '''
    cam_process = Process(target=cam_loop, args=(queue_from_cam,))
    cam_process.start()
    while queue_from_cam.empty():
        pass
    sample_frame = queue_from_cam.get()
    '''

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if ani_multiprocessing == 0:
        # ----------------------------
        # To Play Video: The first frame
        av = Play_AV()
        video_path = BASE_DIR + '/animation/ani01_known_approach.mov'
        audio_enable = 0
        pause = 1
        av.play_av(video_path, pause, audio_enable)
        # ----------------------------
    else:
        av = []



    # ----------------------------
    # Head Pose Detection: by Dlib
    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    hpd = HeadPose(sample_frame, predictor_path)


    # ------------------------------------------
    # Dlib: Load labels_id & face_descriptors of registered faces
    predictor_path = "face_recognition/shape_predictor_5_face_landmarks.dat"
    face_rec_model_path = "face_recognition/dlib_face_recognition_resnet_model_v1.dat"
    fr = FaceRecog(predictor_path, face_rec_model_path, fr_th=0.5)
    iter = 0
    # ------------------------------------------


    # ------------------------------------------
    # Generate a class for event detection such as approach or disappear
    event_detect = Event_Detector()

    # ------------------------
    # Object Tracking by Dlib correlation_tracker
    obj_track = Obj_Tracker()

    # ------------------------
    # Load Database
    filename = BASE_DIR + "/RMI_researchers.csv"
    C_db = Database()
    db = C_db.get_datatbase(filename)

    # --------------------------------
    # Mongo DB
    db_name = "DB_reception"        # define a Database
    coll_name = "RMI_researchers"   # define a Collection
    mgdb = MongoDB(db_name, coll_name)

    db_name = "DB_reception"        # define a Database
    coll_name = "Event"             # define a Collection
    mgdb_event = MongoDB(db_name, coll_name)

    # Test the 'find' of Mongo DB
    #eng_name = "jschoi"
    #name_dict = { "english_name": eng_name }
    #result = mgdb.coll.find(name_dict)
    #pprint.pprint(result[0])
    #print(result[0]["name"])


    # -------------------------------------------------------------
    # chatbot/dialogflow.py  for Dialogflow chatbot platform
    user_key = 'DeepTasK'
    chatbot_id = 'c54e4466-d26d-4966-af1f-ca4d087d0c4a'
    chat = ChatBot(chatbot_id)


    # --------------------------------
    # Create NaverTTS Class
    tts = NaverTTS(0,-1)    # Create a NaverTTS() class from tts/naver_tts.py
    #tts.play("안녕하십니까?")


    # -----------------------
    # Web_API class for web POST
    web_api = Web_API()


    # For instantaneous image capture
    capture_idx = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        '''
        while queue_from_cam.empty():
            pass
        frame = queue_from_cam.get()
        '''

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        max_width = 0   #frame.shape[0]
        max_width_id = -1

        # ---------------------------------
        # Face Recognition
        fr_labels, fr_box, fr_min_dist, max_width_id = fr.face_recognition_iter(iter, frame, obj_track, event_detect)
        iter += 1

        # 아니면, 매번 얼굴인식 수행
        # Face Recognition
        #(fr_labels, fr_box, fr_min_dist) = fr.face_recognition(frame)
        #-------------------------------------


        # ---------------------------------
        # Head Pose Detection for the closed face,
        hpd.head_pose_detection_closed_face(frame, fr_labels, fr_box, max_width_id)

        # ----------------------------
        #   Action Event Detection: action_detection/action_detection.py
        (ad_state, ad_event) = event_detect.approach_disappear(fr_labels, fr_box, max_width_id)


        kor_name = []
        event_name = 'UnknownApproach'
        event_data = {'visitor_name': ""}

        # ----------------------------------------
        # 사용자가 다가온 경우 (그 순간만 수행)
        if ad_event == ACTION_EVENT_APPROACH:
            event_name = 'Approach'     # For query API of Dialogflow
            dialog_flag = True  # Enable dialog when APPROACH, Disable when dialog end   # 대화 종료 시, 카메라 인식을 위해 음성인식을 끈다. -> ACTION_EVENT_APPROACH 이벤트 발생 시 다시 stt_enable = 1로 켠다
            eng_name = fr_labels[max_width_id]      #  인식된 얼굴의 영문 이름 -> csv 파일에서 한국이름을 찾고자 함

            # -----------------------------------------------
            #  To find kor_name from eng_name using MongoDB or csv file
            #kor_name = mgdb.search("english_name", eng_name)
            kor_name = C_db.search(db, "english_name", eng_name)
            # -----------------------------------------------

            video_path = BASE_DIR + '/animation/ani01_known_approach.mov'

            if len(kor_name) > 0:   #  MongoDB/CSV 에서 한국이름을 찾을 수 있는 경우
                # -------------------------------
                # -- Approach 할 때마다 MongoDB에 { "event": "approach", "name": kor_name, "datetime": datetime.datetime.now() } 형태로 기록
                # -------------------------------
                # dict_contents = { "event": "approach", "name": kor_name, "datetime": datetime.datetime.now() }
                # mgdb_event.coll.insert_one(dict_contents)
                # -------------------------------
                dt = []
                for ret in mgdb_event.coll.find({"name": kor_name}):
                    last_time = ret['datetime']
                    now = datetime.datetime.now()
                    dt = now - last_time
                #print(dt)
                #print("days: ", dt.days, ", seconds: ", dt.seconds)


                event_detect.event_label = kor_name
                message = kor_name + "님, 안녕하세요? 반갑습니다."
                try:
                    if dt.days > 0:
                        message = message + " " + str(dt.days) + "일만에 오셨군요."
                        how_long = str(dt.days) + "일"
                    elif dt.seconds > 0:
                        message = message + " " + str(dt.seconds) + "초만에 오셨군요"
                        how_long = str(dt.seconds) + "초"
                    video_path = BASE_DIR + '/animation/ani01_known_approach_memory.mov'
                except Exception as e:
                    pass
                print(message)

                event_data = {'visitor_name': kor_name}

            else:   # MongoDB에서 한국이름을 찾을 수 없는 경우
                message = "안녕하십니까? 저는 안내를 도와드리는 ReceptionBot입니다."
                print(message)

                event_data = {'visitor_name': 'UNKNOWN'}
                video_path = BASE_DIR + '/animation/ani01_unknown_approach.mov'

            # ===============================
            # TTS
            param = {
                'tts_enable': tts_enable,
                'stt_enable': stt_enable,
                'ani_multiprocessing': ani_multiprocessing,
                'ad_event': ad_event,
                'video_path': video_path,
                'audio_enable': 0,
                'pause': 0,
                'video_delay': 100,
                'audio_length': len(message)
            }
            tts_animation(message, tts, av, web_api, gsp, obj_track, param)
            # ===============================


            # -------------------------------
            # Multiprocessing을 시도했으나, cv2.VideoCapture()로 인해 수행이 안됨 -> 확인 필요
            # if tts_enable == 1:
            #    for proc in procs:
            #        proc.join()
            #        procs.pop()
            #    proc = Process(target=tts.play, args=(message,))
            #    procs.append(proc)
            #    proc.start()
            #    print("Proc started! proc: ", proc, "  len(procs): ", len(procs))
            # -------------------------------


            # -------------------------------------------------------------
            # chatbot/dialogflow.py  for Dialogflow chatbot platform
            content = "안녕, 안내를 부탁해요"
            res = chat.get_answer_dialogflow(content, user_key)
            message = res['result']['fulfillment']['speech']
            #    v1 API
            ##event_name = 'Approach'
            ##event_data = {'visitor_name': kor_name}
            # res = chat.event_api_dialogflow(event_name, event_data, user_key)
            # message = res['result']['fulfillment']['speech']
            # -------------------------------------------------------------


            # ===============================
            # TTS
            param = {
                'tts_enable': tts_enable,
                'stt_enable': stt_enable,
                'ani_multiprocessing': ani_multiprocessing,
                'ad_event': ad_event,
                'video_path': BASE_DIR + '/animation/ani01_Person_Place.mov',
                'audio_enable': 0,
                'pause': 0,
                'video_delay': 100,
                'audio_length': len(message)
            }
            tts_animation(message, tts, av, web_api, gsp, obj_track, param)
            # ===============================
        # 사용자가 다가온 경우 (그 순간만 수행) - 끝
        # ----------------------------------------


        # ----------------------------------------
        # 음성 상호작용 하던 사람이 사라진 경우 (그 순간만 수행)
        elif ad_event == ACTION_EVENT_DISAPPEAR:
            if len(event_detect.event_label) > 0:
                message = event_detect.event_label + "님, 안녕히 가세요."
                print(message)
                event_detect.event_label = []

                # ===============================
                # TTS
                param = {
                    'tts_enable': tts_enable,
                    'stt_enable': stt_enable,
                    'ani_multiprocessing': ani_multiprocessing,
                    'ad_event': ad_event,
                    'video_path': BASE_DIR + '/animation/ani01_GoodBye.mov',
                    'audio_enable': 0,
                    'pause': 0,
                    'video_delay': 100,
                    'audio_length': len(message)
                }
                tts_animation(message, tts, av, web_api, gsp, obj_track, param)
                # ===============================

            dialog_flag = False  # Enable dialog when APPROACH, Disable when dialog end   # 대화 종료 시, 카메라 인식을 위해 음성인식을 끈다. -> ACTION_EVENT_APPROACH 이벤트 발생 시 다시 stt_enable = 1로 켠다

        # 상호작용 하던 사람이 사라진 경우 (그 순간만 수행): 끝
        # ----------------------------------------


        # ----------------------------------------
        # 음성 상호작용 시작됨
        elif ad_state == ACTION_STATE_FACE_DETECTED:
            # ---------------------------
            # 음성인식 시작
            try:
                if dialog_flag:
                    if stt_enable == 1:
                        # -------------------------------
                        # STT 재시작
                        #print("------- 음성인식 대기중 -------")
                        gsp.resumeMic()
                        block = False
                        content = gsp.getText(block)
                        if content is not None:
                            print (content)
                        else:
                            # 구글 음성인식기의 경우 1분 제한을 넘으면 None 발생 -> 다시 클래스를 생성시킴
                            print("Recreate Gspeech()!")
                            del gsp
                            gsp = Gspeech()
                    else:
                        q_iter = q_iter + 1
                        dialog_flag = q_iter < q_length
                        content = query[q_iter-1]

            except Exception as e:
                # 음성인식기의 block=False로 해 놓았을 때, 아직 버퍼에 쌓이지 않으면 오류 처리
                content = None
                pass
            # 음성인식 끝
            # ---------------------------


            # --------------------------------------
            # 본격적인 챗봇 구간 시작
            if dialog_flag and content is not None:

                # -------------------------------------------------------------
                # chatbot/dialogflow.py  for Dialogflow chatbot platform
                #    v1 API
                #context_flag = True
                #context_value = "EventApproachHello-followup"
                #res = chat.get_answer_dialogflow(content, user_key, context_flag, context_value)
                res = chat.get_answer_dialogflow(content, user_key)
                message = res['result']['fulfillment']['speech']

                # ===============================
                # TTS
                param = {
                    'tts_enable': tts_enable,
                    'stt_enable': stt_enable,
                    'ani_multiprocessing': ani_multiprocessing,
                    'ad_event': ad_event,
                    'video_path': BASE_DIR + '/animation/ani01_Hi_Short.mov',
                    'audio_enable': 0,
                    'pause': 0,
                    'video_delay': 120,
                    'audio_length': len(message)
                }
                tts_animation(message, tts, av, web_api, gsp, obj_track, param, BASE_DIR + '/animation/ani01_Hi_Short.mov')
                # ===============================

                if (u'끝내자' in content):
                    if len(event_detect.event_label) > 0:
                        message = event_detect.event_label + "님, 안녕히 가세요."
                    else:
                        message = "네. 그럼, 안녕히 가세요."
                    print(message)

                    break

                # --------------------------------------------------
                #   CSV database에서 해당 연구원의 자세한 정보를 가져와 안내함
                try:
                    person_to_visit = res['result']['parameters']['person_to_visit']
                    # ------------------------
                    # 최종석 박사 -> 최종석
                    # 최종석 -> 최종석
                    # ------------------------
                    person_to_visit2 = person_to_visit.split()
                    person_to_visit2 = person_to_visit2[0]
                    #print (person_to_visit2)

                    print('============= print from internal process ==================')
                    # ------------------------
                    # database에 해당 name의 사람이 있으면 그 사람의 information을 갖고 오고,
                    # ''     ''      ''     ''  없으면 ERROR를 갖고 온다.
                    # ------------------------
                    try:
                        info = db[person_to_visit2]
                        try:
                            room_num = info["room#"]
                            message = person_to_visit + "님은 " + room_num + "호 에 계시며, 자세한 정보는 다음과 같습니다."
                        except:
                            message = person_to_visit + "님의 정보는 다음과 같습니다."
                        
                        #info = {
                        #    "name": "최종석",
                        #    "information": {
                        #        "center": "지능로봇연구단",
                        #        "room#": "8402",
                        #        "phone#": "5618",
                        #        "e-mail": "cjs@kist.re.kr"
                        #    }
                        #}
                        
                        # print('   information about ', name, ': ', json.dumps(info, indent=4, ensure_ascii=False))


                    except:
                        message = "죄송합니다만, KIST 국제협력관에서 " + person_to_visit + "님의 정보를 찾을 수 없습니다."
                        message = message + " 찾으시는 다른 분이 계시면 말씀하세요. 끝내시려면 끝내자 라고 해주세요."
                        info = 'ERROR'

                    answer = {
                        'name': person_to_visit2,
                        'information': info
                    }
                    print(message)

                    # ===============================
                    # TTS
                    param = {
                        'tts_enable': tts_enable,
                        'stt_enable': stt_enable,
                        'ani_multiprocessing': ani_multiprocessing,
                        'ad_event': ad_event,
                        'video_path': BASE_DIR + '/animation/ani01_Hi_Short.mov',
                        'audio_enable': 0,
                        'pause': 0,
                        'video_delay': 150,
                        'audio_length': len(message)
                    }
                    tts_animation(message, tts, av, web_api, gsp, obj_track, param,
                                  BASE_DIR + '/animation/ani01_Hi_Short.mov')
                    # ===============================

                    print(json.dumps(answer, indent=4, ensure_ascii=False))
                    #print (info)


                except Exception as e:
                    pass
                #   CSV database에서 해당 연구원의 자세한 정보를 가져와 안내함 끝
                # ------------------------------------------------
            # 본격적인 챗봇 구간 끝
            # --------------------------------------
        # 음성 상호작용 끝
        # ----------------------------------------


        key_in = cv2.waitKey(20) & 0xFF
        if key_in == ord('q'):
            break
        elif key_in == ord('c'):
            f_name = BASE_DIR + "/images/capture" + str(capture_idx) + ".png"
            print("Captured to file: {}".format(f_name))
            capture_idx += 1
            cv2.imwrite(f_name, frame)



        # Display the resulting frame
        winname = "Face Recognition"
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 1280, 10)
        cv2.imshow(winname,frame)   # When Google Speech stt crashes, comment this out!



    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    #cam_process.join()

#----------------------------------------------------
# 메인 함수
#----------------------------------------------------
if __name__ == '__main__':

    stt_enable = 1  # 0: Disable speech recognition (STT), 1: Enable it
    tts_enable = 1  # 0: Disable speech synthesis (TTS),   1: Enable it

    ani_multiprocessing = 1   # 먼저 ./animation 폴더에서  python3 main_server.py 실행시킬 것

    main(stt_enable, tts_enable, ani_multiprocessing)
