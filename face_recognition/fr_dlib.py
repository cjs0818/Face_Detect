# -*- coding: utf-8 -*-

# ----------------------------------------------
# Face recognition: by Dlib
# Camera Capture: by OpenCV
# ----------------------------------------------
'''
"Call this program like this:\n"
"   ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
"You can download a trained facial shape predictor and recognition model from:\n"
"    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
"    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
'''


import numpy as np
import cv2
import pickle    # used for Face recognition by OpenCV

import dlib     # used for Face detection by Dlib
import os
import csv

#import sys
#from skimage import io
#from PIL import Image



class FaceRecog():
    def __init__(self, predictor_path = "./shape_predictor_5_face_landmarks.dat",
                 face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat",
                 fr_th=0.5):
        # ----------------------------
        # Face detection: by Dlib
        self.detector = dlib.get_frontal_face_detector()
        self.fr_th = fr_th

        # ----------------------------
        # Face Recognition: by Dlib
        #predictor_path = "./shape_predictor_5_face_landmarks.dat"
        #face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
        self.sp = dlib.shape_predictor(predictor_path)
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        self.face_descriptor_p = []

        try:
            (self.label_ids, self.fd_known) = self.load_registered_face()
        except:
            (self.label_ids, self.fd_known) = self.save_registered_face()


    def save_registered_face(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 이 파일이 있는 폴더 위치
        PARENT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."   # images 폴더가 있는 위치
        image_dir = os.path.join(PARENT_DIR, "images")

        current_id = 0
        label_ids = []

        each_label_cnt = 0
        face_descriptor_sum = np.zeros(128)
        fd_known = []

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg") or file.endswith("PNG") or file.endswith("JPG"):
                    path = os.path.join(root, file)
                    label = os.path.basename(root).replace(" ", "-").lower()
                    #print(label, ", ", path)


                    with open(path, 'rb') as image_file:

                        #frame = Image.open(image_file)
                        frame = cv2.imread(path, cv2.IMREAD_COLOR)
                        frame = cv2.resize(frame, (320, 320))

                        # ---------------------------------
                        # Recognize by Dlib

                        # Ask the detector to find the bounding boxes of each face. The 1 in the
                        # second argument indicates that we should upsample the image 1 time. This
                        # will make everything bigger and allow us to detect more faces.
                        dets = self.detector(frame, 1)
                        #print("Number of faces detected: {}".format(len(dets)))

                        for k, d in enumerate(dets):
                            each_label_cnt += 1

                            # Get the landmarks/parts for the face in box d.
                            shape = self.sp(frame, d)

                            # Compute the 128D vector that describes the face in img identified by
                            # shape.  In general, if two face descriptor vectors have a Euclidean
                            # distance between them less than 0.6 then they are from the same
                            # person, otherwise they are from different people. Here we just print
                            # the vector to the screen.
                            face_descriptor = self.facerec.compute_face_descriptor(frame, shape)
                            face_descriptor_sum = np.add(face_descriptor_sum, face_descriptor)


                        if not label in label_ids.values():
                            if(current_id > 0):
                                if(each_label_cnt > 0):
                                    #print("(current_id, each_label_cnt) = (%2d, %2d)" % (current_id, each_label_cnt))
                                    fd_avg = np.divide(face_descriptor_sum, each_label_cnt)
                                    fd_known.append(fd_avg)
                                else:
                                    label_ids.popitem()
                                    current_id -= 1

                            label_ids[current_id] = label
                            current_id += 1
                            each_label_cnt = 0
                            face_descriptor_sum = np.zeros(128)
        if(each_label_cnt > 0):
            fd_known.append(np.divide(face_descriptor_sum, each_label_cnt))
        else:
            label_ids.popitem()

        print(label_ids)


        # File Write 'lables_ids' & 'fd_known' into a pickle file
        pickle_file = BASE_DIR + "/face-labels.pickle"
        with open(pickle_file, 'wb') as f:
            pickle.dump(label_ids, f)
            pickle.dump(fd_known, f)

        #print(len(fd_known))
        #print(label_ids)

        return (label_ids, fd_known)


    def load_registered_face(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        pickle_file = BASE_DIR + "/face-labels.pickle"

        with open(pickle_file, 'rb') as f:
            label_ids = pickle.load(f)
            fd_known = pickle.load(f)


        print("Registered_Faces: Load Completed !!!")
        print(label_ids)
        #print(len(fd_known))

        return (label_ids, fd_known)


    def face_recognition(self, frame):
        # ---------------------------------
        # Face Detection

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = self.detector(frame, 1)
        #print("Number of faces detected: {}".format(len(dets)))

        fr_labels = []
        fr_box = []
        fr_min_dist = []

        for k, d in enumerate(dets):
            #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #    k, d.left(), d.top(), d.right(), d.bottom()))

            # Draw ROI box for each face found by face detection
            color = (255, 0, 0)  # BGR 0-255
            x = d.left()
            y = d.top()
            end_cord_x = d.right()
            end_cord_y = d.bottom()
            stroke = 2
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

            # ---------------------------------
            # Recognize by Dlib

            # Get the landmarks/parts for the face in box d.
            shape = self.sp(frame, d)

            # Compute the 128D vector that describes the face in img identified by
            # shape.  In general, if two face descriptor vectors have a Euclidean
            # distance between them less than 0.6 then they are from the same
            # person, otherwise they are from different people. Here we just print
            # the vector to the screen.
            face_descriptor = self.facerec.compute_face_descriptor(frame, shape)

            # self.fr_th = 0.5
            #print(self.label_ids)
            # print(len(fd_known))

            min_dist = self.fr_th
            selected_label = None

            for id in self.label_ids.keys():
                dist = np.subtract(face_descriptor, self.fd_known[id])
                dist = np.sqrt(np.dot(dist, dist))
                #print("id: %2d, dist: %4.2f" % (id, dist))
                if (dist < self.fr_th and dist < min_dist):
                    selected_label = self.label_ids[id]
                    min_dist = dist

            if (selected_label != None):
                fr_labels.append(selected_label)
                fr_box.append(d)
                fr_min_dist.append(min_dist)

        return(fr_labels, fr_box, fr_min_dist)


    def face_recognition_indiv(self, frame):

        fr_labels = []
        fr_box = []
        fr_min_dist = []


        # ---------------------------------
        # Recognize by Dlib

        # Get the landmarks/parts for the face in box d.
        shape = self.sp(frame, d)

        # Compute the 128D vector that describes the face in img identified by
        # shape.  In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people. Here we just print
        # the vector to the screen.
        face_descriptor = self.facerec.compute_face_descriptor(frame, shape)

        # self.fr_th = 0.5
        #print(self.label_ids)
        #print(len(fd_known))

        min_dist = self.fr_th
        selected_label = None

        for id in self.label_ids.keys():
            dist = np.subtract(face_descriptor, self.fd_known[id])
            dist = np.sqrt(np.dot(dist, dist))
            #print("id: %2d, dist: %4.2f" % (id, dist))
            if (dist < self.fr_th and dist < min_dist):
                selected_label = self.label_ids[id]
                min_dist = dist

        if (selected_label != None):
            fr_labels.append(selected_label)
            fr_box.append(d)
            fr_min_dist.append(min_dist)

        return(fr_labels, fr_box, fr_min_dist)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)

    #------------------------------------------
    # Dlib: Load labels_id & face_descriptors of registered faces
    predictor_path = "./shape_predictor_5_face_landmarks.dat"
    face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
    fr = FaceRecog(predictor_path, face_rec_model_path, fr_th=0.5)
    (labels, fd_known) = fr.load_registered_face()
    #------------------------------------------



    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------------------------
        # Face Recognition
        (fr_labels, fr_box, fr_min_dist) = fr.face_recognition(frame)

        for id in range(len(fr_labels)):
            selected_label = fr_labels[id]
            d = fr_box[id]
            min_dist = fr_min_dist[id]

            print(selected_label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            conf_str = "{0:3.1f}".format(min_dist)
            name = selected_label + ", [" + conf_str + "]"
            color = (255, 255, 255)
            stroke = 1  # 글씨 굵기 ?
            cv2.putText(frame, name, (d.left(), d.top()), font, 0.5, color, stroke, cv2.LINE_AA)


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
