# -*- coding: utf-8 -*-

# ----------------------------------------------
# Face detection, Head Pose detection, and Face recognition: by Dlib
# Camera Capture: by OpenCV
# ----------------------------------------------

import numpy as np
import cv2
import pickle    # used for Face recognition by OpenCV


import dlib     # used for Face detectiob by Dlib
import os
#import sys
#from skimage import io
#from PIL import Image


# ----------------------------
# Face detection: by Dlib
detector = dlib.get_frontal_face_detector()

# ----------------------------
# Face Recognition: by Dlib
#  ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
#    You can download a trained facial shape predictor and recognition model from:\n"
#     http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
#     http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
from face_recognition.fr_dlib import FaceRecog

# ----------------------------
# Head Pose detection: by Dlib
#   You can download a trained facial shape predictor from:
#    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
from hp_detection import HeadPose



'''
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
'''


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
    (labels, fd_known) = fr.load_registered_face()
    # ------------------------------------------



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


        # ---------------------------------
        # Face Recognition
        (fr_labels, fr_box, fr_min_dist) = fr.face_recognition(frame)

        for id in range(len(fr_labels)):
            selected_label = fr_labels[id]
            d = fr_box[id]
            min_dist = fr_min_dist[id]

            if(selected_label != None):
                print(selected_label)
                font = cv2.FONT_HERSHEY_SIMPLEX
                conf_str = "{0:3.1f}".format(min_dist)
                name = selected_label + ", [" + conf_str + "]"
                color = (255, 255, 255)
                stroke = 1   # 글씨 굵기 ?
                cv2.putText(frame, name, (d.left(), d.top()), font, 0.5, color, stroke, cv2.LINE_AA)


            # ---------------------------------
            # Get the landmarks/parts for the face in box d.
            shape = hpd.predictor(frame, d)
            #print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

            # Find Head Pose using the face landmarks and Draw them on the screen.
            (p1, p2) = hpd.draw_landmark_headpose(frame, shape)
            roi_ratio = (d.right() - d.left()) / frame.shape[0]

            dist = np.subtract(p2, p1)
            dist = np.sqrt(np.dot(dist, dist))
            dist_ratio = dist / (d.right() - d.left())
            print((roi_ratio, dist_ratio))

            roi_ratio_th = 0.15
            dist_ratio_th = 0.75  # 0.03
            print(" ")
            print("roi_ratio: %3.2f, dist_ratio: %5.4f" % (roi_ratio, dist_ratio))
            if roi_ratio > roi_ratio_th and dist_ratio < dist_ratio_th:
                cv2.line(frame, p1, p2, (0, 0, 255), 2)
            else:
                cv2.line(frame, p1, p2, (0, 255, 0), 2)



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