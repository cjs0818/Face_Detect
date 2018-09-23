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


import dlib
import os
import sys
#from skimage import io
#from PIL import Image

import csv


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
from action_detection.action_detection import Event_Detector
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


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    ret, sample_frame = cap.read()

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
    # ------------------------------------------


    # ------------------------------------------
    # Generate a class for event detection such as approach or disappear
    event_detect = Event_Detector()

    # ------------------------
    # Object Tracking by Dlib correlation_tracker
    obj_track = Obj_Tracker()

    # ------------------------
    # Load Database
    #PARENT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
    #DB_DIR = os.path.join(PARENT_DIR, "Receptionbot_Danbee/receptionbot")
    #filename = DB_DIR + "/RMI_researchers.csv"
    filename = os.path.dirname(os.path.abspath(__file__)) + "/RMI_researchers.csv"
    db = get_datatbase(filename)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
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
        # ---------------------------------
        # Face Recognition
        (fr_labels, fr_box, fr_min_dist) = fr.face_recognition(frame)


        # --------------------------------------
        # Object Tracking for undetected face
        if obj_track.track_started == False:
            if len(fr_labels) > 0:
                obj_track.track_started = True
                obj_track.start_tracking(frame, fr_box[max_width_id])
                obj_track.label = fr_labels[max_width_id]
                obj_track.tracking(frame)

        else:
            if len(fr_labels) > 0 and fr_labels[max_width_id] == obj_track.label:
                obj_track.start_tracking(frame, fr_box[max_width_id])
            else:
                #obj_track.track_running = True
                max_width_id = 0
                fr_labels = []
                fr_box = []
                fr_min_dist = []
                fr_labels.append(obj_track.label)
                fr_box.append(obj_track.roi)
                fr_min_dist.append(0)

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
            shape = hpd.predictor(frame, d)
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

        kor_name = []
        if ad_event == ACTION_EVENT_APPROACH:
            eng_name = fr_labels[max_width_id]
            #print(db.keys())
            for name in db.keys():
                info = db[name]
                if info["english_name"] == eng_name:
                    kor_name = name
            if len(kor_name) > 0:
                if ad_event == ACTION_EVENT_APPROACH:
                    event_detect.event_label = kor_name
                    print("Hi! Nice to meet you, {}".format(kor_name))
        elif ad_event == ACTION_EVENT_DISAPPEAR:
            if len(event_detect.event_label) > 0:
                print("Good Bye! {}".format(event_detect.event_label))
                event_detect.event_label = []



        # Display the resulting frame
        cv2.imshow('frame',frame)

        #win.add_overlay(dets)


        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


#----------------------------------------------------
# 메인 함수
#----------------------------------------------------
if __name__ == '__main__':
    main()
