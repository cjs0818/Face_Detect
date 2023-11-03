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
from gtts import gTTS


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

# ------------------------
# Thread (thread_chatbot)
import threading
from threading import Lock

from playsound import playsound
from pydub import AudioSegment

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

class FI():  # Face Interaction Class
    def __init__(self, cap):

        cap.set(3, 320)
        cap.set(4, 240)

        ret, sample_frame = cap.read()

        self.cap = cap
        # ----------------------------
        # Head Pose Detection: by Dlib
        predictor_path = "./shape_predictor_68_face_landmarks.dat"
        #predictor = dlib.shape_predictor(predictor_path)
        self.hpd = HeadPose(sample_frame, predictor_path)

        # ------------------------------------------
        # Dlib: Load labels_id & face_descriptors of registered faces
        predictor_path = "face_recognition/shape_predictor_5_face_landmarks.dat"
        face_rec_model_path = "face_recognition/dlib_face_recognition_resnet_model_v1.dat"
        self.fr = FaceRecog(predictor_path, face_rec_model_path, fr_th=0.5)
        self.iter = 0
        # ------------------------------------------

        # ------------------------------------------
        # Generate a class for event detection such as approach or disappear
        self.event_detect = Event_Detector()

        # ------------------------
        # Object Tracking by Dlib correlation_tracker
        self.obj_track = Obj_Tracker()

        self.max_width = 0  #frame.shape[0]
        self.max_width_id = -1
        self.fr_labels = []
        self.fr_box = []
        self.fr_min_dist = 0
        self.ad_state = 0
        self.ad_event = 0

    def run(self):
        # Capture frame-by-frame
        cap = self.cap

        hpd = self.hpd
        fr = self.fr
        event_detect = self.event_detect
        obj_track = self.obj_track

        iter = self.iter
        max_width = self.max_width
        max_width_id = self.max_width_id
        fr_labels = self.fr_labels
        fr_box = self.fr_box
        fr_min_dist = self.fr_min_dist
        ad_state = self.ad_state
        ad_event = self.ad_event

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
            #   Select the closest face
            d_width = d.right() - d.left()
            if(d_width > max_width):
                max_width_id = id
                max_width = d_width


        if(len(fr_labels) > 0):
            # ---------------------------------
            # Head Pose Detection for the closest face,

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

        self.iter = iter
        self.fr_labels = fr_labels
        self.fr_box = fr_box
        self.fr_min_dist = fr_min_dist
        self.max_max_width = max_width
        self.max_width_id = max_width_id
        self.ad_state = ad_state
        self.ad_event = ad_event

        return frame



def Thread_TTS(q, tts):
    while True:
        if q.empty() is False:
            message = q.get()
            print(f"message: {message}, tts.engine.isBusy(): {tts.engine.isBusy()}")
            tts.play(message)
            while tts.engine.isBusy():
                time.sleep(0.01)
        else:
            pass
    
class C_ChatBot(): #(stt_enable, tts_enable, gsp, fi, chat, db, tts, mgdb, mgdb_event):
    def __init__(self, stt_enable, tts_enable, gsp, fi, chat, db, tts, mgdb, mgdb_event):
        self.stt_enable = stt_enable
        self.tts_enable = tts_enable
        self.gsp = gsp
        self.fi = fi
        self.chat = chat
        self.db = db
        self.tts = tts
        self.mgdb = mgdb
        self.mgdb_event = mgdb_event

        self.queue = Queue()
        thread_tts = threading.Thread(target=Thread_TTS, args=(self.queue,tts,))
        #thread_tts.daemon = True
        thread_tts.start()

         
    def run(self):
        fi = self.fi
        gsp = self.gsp
        db = self.db
        tts = self.tts
        chat = self.chat

        project_id = "receptionbot-3b113"
        session_id = "your-session-id"
        language_code = 'ko-KR'  # a BCP-47 language tag
        
        cnt = 0
        while True:
            if cnt < 3:
                message = f"테스트입니다 {cnt}"
                self.queue.put(message)
                cnt = cnt + 1
            else:
                break
        print(f"cnt: {cnt}")




def main(stt_enable=1, tts_enable=1):
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


    cap = cv2.VideoCapture(0)


    fi = FI(cap)    # Face Interaction Class

    '''
    cam_process = Process(target=cam_loop, args=(queue_from_cam,))
    cam_process.start()
    while queue_from_cam.empty():
        pass
    sample_frame = queue_from_cam.get()
    '''


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


    chat = ChatBot()

    #-----------------------------------------------------
    c_chatbot = C_ChatBot(stt_enable, tts_enable, gsp, fi, chat, db, tts, mgdb, mgdb_event)
    c_chatbot.run()

    # Multi-processing
    procs = []

    # For instantaneous image capture
    capture_idx = 0

    while(True):
        # Capture frame-by-frame
        frame = fi.run()

        

        #time.sleep(0.01)
        # -------------------------------
        # STT 재시작
        #if stt_enable == 1 and tts_enable == 1:
        #    gsp.resumeMic()


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
