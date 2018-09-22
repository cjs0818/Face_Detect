# -*- coding: utf-8 -*-

# ----------------------------------------------
# Save Face Descriptor for Face recognition: by Dlib
# ----------------------------------------------



import numpy as np
import cv2
import pickle    # used for Face recognition by OpenCV


import dlib     # used for Face detection by Dlib
import os


def save_face_descriptor(image_dir):

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    each_label_cnt = 0
    face_descriptor_sum = np.zeros(128)
    fd_known = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("PNG") or file.endswith("JPG"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                # print(label, ", ", path)

                with open(path, 'rb') as image_file:

                    # frame = Image.open(image_file)
                    frame = cv2.imread(path, cv2.IMREAD_COLOR)
                    frame = cv2.resize(frame, (320, 320))

                    # ---------------------------------
                    # Recognize by Dlib

                    # Ask the detector to find the bounding boxes of each face. The 1 in the
                    # second argument indicates that we should upsample the image 1 time. This
                    # will make everything bigger and allow us to detect more faces.
                    dets = self.detector(frame, 1)
                    # print("Number of faces detected: {}".format(len(dets)))

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
                        if (current_id > 0):
                            if (each_label_cnt > 0):
                                # print("(current_id, each_label_cnt) = (%2d, %2d)" % (current_id, each_label_cnt))
                                fd_avg = np.divide(face_descriptor_sum, each_label_cnt)
                                fd_known.append(fd_avg)
                            else:
                                label_ids.popitem()
                                current_id -= 1

                        label_ids[current_id] = label
                        current_id += 1
                        each_label_cnt = 0
                        face_descriptor_sum = np.zeros(128)
    if (each_label_cnt > 0):
        fd_known.append(np.divide(face_descriptor_sum, each_label_cnt))
    else:
        label_ids.popitem()

    print(label_ids)


#----------------------------------------------------
# 메인 함수
#----------------------------------------------------
if __name__ == '__main__':

    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."  # images 폴더가 있는 위치
    image_dir = os.path.join(BASE_DIR, "images")
    save_face_descriptor(image_dir)
