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
#from skimage import io
#from PIL import Image

import csv

# Mongo DB
from pymongo import MongoClient
import datetime
import pprint

import threading
from threading import Lock


#-------------------------------------------------------------
# chatbot/dialogflow.py  for Dialogflow chatbot platform
#from chatbot.dialogflow import ChatBot   # Chatbot platform: Dialogflow.ai
from chatbot.dialogflow_v2 import ChatBot   # Chatbot platform: Dialogflow.ai
from stt.gspeech import Gspeech     # STT: Google Cloud Speech
import json

# ----------------------------------------------------------
#  You need to setup PYTHONPATH to include tts/naver_tts.py
#   ex.) export PYTHONPATH=/Users/jschoi/work/ChatBot/Receptionbot_Danbee/receptionbot:$PYTHONPATH
#from tts.naver_tts import NaverTTS  # TTS: NaverTTS
from tts.pyttsx3_tts import PyTTSX3  # TTS: NaverTTS


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


# ----------------------------------------------------
# database from CSV file
# ----------------------------------------------------
def get_datatbase(filename):
    #filename = 'RMI_researchers.csv'

    with open(filename, 'r', encoding='UTF-8-sig') as f:
        csv_data = csv.reader(f, delimiter=',')
        print("-------------")
        dict = {}
        row_cnt = 0
        for row in csv_data:
            row_cnt = row_cnt + 1
            if row_cnt == 1:
                key = row
            else:
                for i in range(0, len(row), 1):
                    if i == 0:
                        # print(dict_name)
                        dict_info = {}
                    else:
                        dict_info.update({key[i]: row[i]})
                        # print(dict_info)
                dict.update({row[0]: dict_info})
                # print("dict_name = ", dict_name)

    # json_data = json.dumps(dict, indent=4, ensure_ascii=False)
    # print(json_data)

    return dict


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

# Mongo DB
#    folder: BASE_DIR/data/db
class MongoDB():
    def __init__(self, db_name="DB_reception", coll_name="RMI_researchers"):
        #self.db_client = MongoClient('localhost', 27017)
        self.db_client = MongoClient('mongodb+srv://pristine70:cjsy6918@atlascluster.zytuhxo.mongodb.net/')
        #self.db = self.db_client["DB_Episode"]
        #self.coll = self.db.coll_Test
        self.db = self.db_client[db_name]
        self.coll = self.db[coll_name]

    def insert(self, post):
        post_id = self.mdb_collection.insert_one(post).inserted_id
        coll_list = self.mdb.collection_names()
        print(coll_list)


def loop_chatbot(stt_enable=1, tts_enable=1):
    global ad_state, ad_event, fr_labels, max_width_id
    global event_detect

    if stt_enable == 1:
        dialog_flag = True  # Enable speech recognition when APPROACH, Disable when dialog ends
        gsp = Gspeech()
    else:
        # 음성인식 아닌 경우, 테스트 query에 대해 문장 단위로 테스트
        query = [
            "사람",
                "아나스타샤를 찾으러 왔어요",
            "안녕, 안내를 부탁해요",
            "사람",
                "최종석 박사님을 만나러 왔어요",
            "안녕, 안내를 부탁해요",
            "사람",
                "홍길동님을 찾으러 왔어요",
            "안녕, 안내를 부탁해요",
            "사람",
                "여진구 박사님이요",
                "끝내자"
                 ]
        q_length = len(query)
        q_iter = 0
        dialog_flag = q_iter < q_length
    # ------------------------
    # Load Database
    #PARENT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
    #DB_DIR = os.path.join(PARENT_DIR, "Receptionbot_Danbee/receptionbot")
    #filename = DB_DIR + "/RMI_researchers.csv"
    filename = os.path.dirname(os.path.abspath(__file__)) + "/RMI_researchers.csv"
    db = get_datatbase(filename)

    # --------------------------------
    # Create NaverTTS Class
    #tts = NaverTTS(0,-1)    # Create a NaverTTS() class from tts/naver_tts.py
    tts = PyTTSX3()
    #tts.play("안녕하십니까?")

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
    #user_key = 'DeepTasK'
    #chatbot_id = 'c54e4466-d26d-4966-af1f-ca4d087d0c4a'
    #chat = ChatBot(chatbot_id)

    project_id = "receptionbot-3b113"
    session_id = "your-session-id"
    language_code = 'ko-KR'  # a BCP-47 language tag
    chat = ChatBot()


    ad_event = []
    ad_state = []
   
    while(True):
        
        kor_name = []
        event_name = 'UnknownApproach'
        event_data = {'visitor_name': ""}
        
        if ad_event == ACTION_EVENT_APPROACH:
            event_name = 'Approach'     # For query API of Dialogflow
            dialog_flag = True  # Enable dialog when APPROACH, Disable when dialog end   # 대화 종료 시, 카메라 인식을 위해 음성인식을 끈다. -> ACTION_EVENT_APPROACH 이벤트 발생 시 다시 stt_enable = 1로 켠다
            eng_name = fr_labels[max_width_id]      #  인식된 얼굴의 영문 이름 -> csv 파일에서 한국이름을 찾고자 함

            #------------------------
            # Search from MongoDB
            #name_dict = {"english_name": eng_name}
            #result = mgdb.coll.find(name_dict)
            #try:
            #    kor_name = result[0]["name"]
            #except Exception as e:
            #    pass

            # ----------------------------
            # -- Search from csv file
            for name in db.keys():
                info = db[name]
                if info["english_name"] == eng_name:
                    kor_name = name


            if len(kor_name) > 0:   #  MongoDB에서 한국이름을 찾을 수 있는 경우
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
                except Exception as e:
                    pass
                print(message)

                event_data = {'visitor_name': kor_name}

            else:   # MongoDB에서 한국이름을 찾을 수 없는 경우
                message = "안녕하십니까? 저는 안내를 도와드리는 ReceptionBot입니다."
                print(message)

                event_data = {'visitor_name': 'UNKNOWN'}

            # ===============================
            # TTS
            if tts_enable == 1:
                if stt_enable == 1:  # TTS 하는 동안 STT 일시 중지 --
                    gsp.pauseMic()
                tts.play(message)
                while tts.engine.isBusy():
                    time.sleep(0.01)
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
            #    v1 API
            ##event_name = 'Approach'
            ##event_data = {'visitor_name': kor_name}
            #res = chat.event_api_dialogflow(event_name, event_data, user_key)
            #message = res['result']['fulfillment']['speech']
            content = "안녕, 안내를 부탁해요"
            #res = chat.get_answer_dialogflow(content, user_key)
            #message = res['result']['fulfillment']['speech']
            
            message = chat.detect_intent(project_id, session_id, content, language_code)

            #print(tts.engine.isBusy())

            # ===============================
            # TTS
            if tts_enable == 1:
                if stt_enable == 1:  # TTS 하는 동안 STT 일시 중지 --
                    gsp.pauseMic()
                tts.play(message)
            # -------------------------------


        elif ad_event == ACTION_EVENT_DISAPPEAR:
            if len(event_detect.event_label) > 0:
                message = event_detect.event_label + "님, 안녕히 가세요."
                print(message)
                event_detect.event_label = []
                # ===============================
                # TTS
                if tts_enable == 1:
                    if stt_enable == 1:  # TTS 하는 동안 STT 일시 중지 --
                        gsp.pauseMic()
                    tts.play(message)
            dialog_flag = False

        elif ad_state == ACTION_STATE_FACE_DETECTED:
            # 음성인식

            #dialog_flag = False

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

            #dialog_flag = False

            if dialog_flag and content is not None:
                if (u'끝내자' in content):
                    if len(event_detect.event_label) > 0:
                        message = event_detect.event_label + "님, 안녕히 가세요."
                    else:
                        message = "네. 안녕히 가세요."
                    print(message)
                    # ===============================
                    if tts_enable == 1:
                        if stt_enable == 1:  # TTS 하는 동안 STT 일시 중지 --
                            gsp.pauseMic()
                        tts.play(message)
                    # -------------------------------
                    break

                # -------------------------------------------------------------
                # chatbot/dialogflow.py  for Dialogflow chatbot platform
                #    v1 API
                #context_flag = True
                #context_value = "EventApproachHello-followup"
                #res = chat.get_answer_dialogflow(content, user_key, context_flag, context_value)
                
                #res = chat.get_answer_dialogflow(content, user_key)
                #message = res['result']['fulfillment']['speech']

                message = chat.detect_intent(project_id, session_id, content, language_code)

                # ===============================
                if tts_enable == 1:
                    if stt_enable == 1:  # TTS 하는 동안 STT 일시 중지 --
                        gsp.pauseMic()
                    tts.play(message)
                # -------------------------------

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


                        # This is the end of a dialog
                        #dialog_flag = False  # Enable dialog when APPROACH, Disable when dialog end   # 대화 종료 시, 카메라 인식을 위해 음성인식을 끈다. -> ACTION_EVENT_APPROACH 이벤트 발생 시 다시 stt_enable = 1로 켠다
                        #if stt_enable == 1:
                        #    gsp.pauseMic()
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
                    if tts_enable == 1:
                        if stt_enable == 1:  # TTS 하는 동안 STT 일시 중지 --
                            gsp.pauseMic()
                        tts.play(message)
                    # -------------------------------
                    print(json.dumps(answer, indent=4, ensure_ascii=False))
                    #print (info)


                except Exception as e:
                    pass

        time.sleep(0.01)
        # -------------------------------
        # STT 재시작
        #if stt_enable == 1 and tts_enable == 1:
        #    gsp.resumeMic()

def main(stt_enable=1, tts_enable=1):



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

    # ----------------------------
    # Head Pose Detection: by Dlib
    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    hpd = HeadPose(sample_frame, predictor_path)


    # ------------------------------------------
    # Dlib: Load labels_id & face_descriptors of registered faces
    predictor_path = "face_recognition/shape_predictor_5_face_landmarks.dat"
    face_rec_model_path = "face_recognition/dlib_face_recognition_resnet_model_v1.dat"
    fr = FaceRecog(predictor_path, face_rec_model_path, fr_th=0.5)
    iter = 0
    # ------------------------------------------

    #-----------------------------------------------------
    thread_chatbot = threading.Thread(target=loop_chatbot, args=(stt_enable, tts_enable))
    thread_chatbot.daemon = True
    thread_chatbot.start()
    #thread_chatbot.join()

    global ad_state, ad_event, fr_labels, max_width_id
    global event_detect

    # ------------------------------------------
    # Generate a class for event detection such as approach or disappear
    event_detect = Event_Detector()

    # ------------------------
    # Object Tracking by Dlib correlation_tracker
    obj_track = Obj_Tracker()


    # Multi-processing
    procs = []

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


        '''
        # ---------------------------------
        # Face Detection
    
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        
        dets = detector(frame, 1)
        print("Number of faces detected: {}".format(len(dets)))
    
    
        for k, d in enumerate(dets):
    
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
    
            # Draw ROI box for each face found by face detection
            color = (255, 0, 0)  # BGR 0-255
            x = d.left()
            y = d.top()
            end_cord_x = d.right()
            end_cord_y = d.bottom()
            stroke = 2
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        '''

        max_width = 0   #frame.shape[0]
        max_width_id = -1



        #-------------------------------------
        #  일정시간마다 tracking reset하기
        if iter % 100 == 0:
            obj_track.track_started = False
            if obj_track.track_started == True:
                if len(fr_labels) > 0 and fr_labels[max_width_id] == "unknown_far":
                    event_detect.reset()
        iter += 1

        # 아니면, 매번 얼굴인식 수행
        # Face Recognition
        #(fr_labels, fr_box, fr_min_dist) = fr.face_recognition(frame)
        #-------------------------------------

        # --------------------------------------
        # Object Tracking for undetected face
        #   Ref: https://www.codesofinterest.com/2018/02/track-any-object-in-video-with-dlib.html
        if obj_track.track_started == False:
            # ---------------------------------
            # Face Recognition
            (fr_labels, fr_box, fr_min_dist) = fr.face_recognition(frame)

            if len(fr_labels) > 0:
                obj_track.track_started = True
                obj_track.start_tracking(frame, fr_box[max_width_id])
                obj_track.label = fr_labels[max_width_id]
                obj_track.tracking(frame)
                obj_track.min_dist = fr_min_dist[max_width_id]

        else:
            if len(fr_labels) > 0 and fr_labels[max_width_id] == "unknown_far":
                event_detect.reset()

                # ---------------------------------
                # Face Recognition
                (fr_labels, fr_box, fr_min_dist) = fr.face_recognition(frame)

                if len(fr_labels) > 0:
                    obj_track.track_started = True
                    obj_track.start_tracking(frame, fr_box[max_width_id])
                    obj_track.label = fr_labels[max_width_id]
                    obj_track.tracking(frame)
            else:
                max_width_id = 0
                fr_labels = []
                fr_box = []
                fr_min_dist = []
                fr_labels.append(obj_track.label)
                fr_box.append(obj_track.roi)
                fr_min_dist.append(obj_track.min_dist)

                obj_track.tracking(frame)
                if obj_track.track_started == False:
                    fr_labels = []
                    fr_box = []
        # --------------------------------------


        # --------------------------------------
        # Display for the name of the selected face
        for id in range(len(fr_labels)):
            selected_label = fr_labels[id]
            d = fr_box[id]
            min_dist = fr_min_dist[id]

            if(selected_label != None):
                #print(selected_label)
                font = cv2.FONT_HERSHEY_SIMPLEX
                conf_str = "{0:3.1f}".format(min_dist)
                name = selected_label + ", [" + conf_str + "]"
                color = (255, 255, 255)
                stroke = 1   # 글씨 굵기 ?
                cv2.putText(frame, name, (d.left(), d.top()), font, 0.5, color, stroke, cv2.LINE_AA)


            # ---------------------------------
            #   Select the closed face
            d_width = d.right() - d.left()
            if(d_width > max_width):
                max_width_id = id

        if(len(fr_labels) > 0):
            # ---------------------------------
            # Head Pose Detection for the closed face,

            # Get the landmarks/parts for the face in box d.
            d = fr_box[max_width_id]
            shape = hpd.predictor(frame, d)    # predict 68_face_landmarks
            #print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

            # Find Head Pose using the face landmarks and Draw them on the screen.
            (p1, p2) = hpd.draw_landmark_headpose(frame, shape)
            roi_ratio = (d.right() - d.left()) / frame.shape[0]

            dist = np.subtract(p2, p1)
            dist = np.sqrt(np.dot(dist, dist))
            dist_ratio = dist / (d.right() - d.left())

            roi_ratio_th = 0.15
            dist_ratio_th = 0.75  # 0.03
            #print(" ")
            #print("roi_ratio: %3.2f, dist_ratio: %5.4f" % (roi_ratio, dist_ratio))
            if roi_ratio > roi_ratio_th and dist_ratio < dist_ratio_th:
                cv2.line(frame, p1, p2, (0, 0, 255), 2)
            else:
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

        (ad_state, ad_event) = event_detect.approach_disappear(fr_labels, fr_box, max_width_id)





        # Display the resulting frame
        cv2.imshow('frame',frame)   # When Google Speech stt crashes, comment this out!

        #win.add_overlay(dets)

        key_in = cv2.waitKey(20) & 0xFF
        if key_in == ord('q'):
            break
        elif key_in == ord('c'):
            f_name = "./images/capture" + str(capture_idx) + ".png"
            print("Captured to file: {}".format(f_name))
            #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            #f_name = BASE_DIR + f_name
            capture_idx += 1
            cv2.imwrite(f_name, frame)


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

    main(stt_enable, tts_enable)
