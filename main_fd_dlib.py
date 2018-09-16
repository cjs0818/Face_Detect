# ----------------------------------------------
# Face detection, Head Pose detection, and Face recognition: by Dlib
# Camera Capture: by OpenCV
# ----------------------------------------------

import numpy as np
import cv2
import pickle    # used for Face recognition by OpenCV

import sys
import dlib     # used for Face detectiob by Dlib
from skimage import io
import os
from PIL import Image

# Head Pose Estimator from https://github.com/yinguobing/head-pose-estimation
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

# ----------------------------
# Face detection: by Dlib
detector = dlib.get_frontal_face_detector()

# ----------------------------
# Head Pose detection: by Dlib
predictor_path = "./shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
'''
You can download a trained facial shape predictor from:
    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
'''

# ----------------------------
# Face Recognition: by Dlib
predictor_path = "./shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
face_descriptor_p = []
'''
"Call this program like this:\n"
"   ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
"You can download a trained facial shape predictor and recognition model from:\n"
"    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
"    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
'''


nose_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1) for _ in range(6)]



'''
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
'''

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

# ----------------------------
# Head Pose Estimator from https://github.com/yinguobing/head-pose-estimation
# Introduce pose estimator to solve pose. Get one frame to setup the
# estimator according to the image size.
_, sample_frame = cap.read()
height, width = sample_frame.shape[:2]
pose_estimator = PoseEstimator(img_size=(height, width))

# Introduce scalar stabilizers for pose.
pose_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1) for _ in range(6)]

# Introduce mark_detector to detect landmarks.
mark_detector = MarkDetector()


# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner

])


def load_registered_face():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")

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
                    dets = detector(frame, 1)
                    #print("Number of faces detected: {}".format(len(dets)))

                    for k, d in enumerate(dets):
                        each_label_cnt += 1

                        # Get the landmarks/parts for the face in box d.
                        shape = sp(frame, d)

                        # Compute the 128D vector that describes the face in img identified by
                        # shape.  In general, if two face descriptor vectors have a Euclidean
                        # distance between them less than 0.6 then they are from the same
                        # person, otherwise they are from different people. Here we just print
                        # the vector to the screen.
                        face_descriptor = facerec.compute_face_descriptor(frame, shape)
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
    print(len(fd_known))

    #print(label_ids)


    return (label_ids, fd_known)



def draw_landmark_headpose(frame, shape):
    """Get marks ready for pose estimation from 68 marks"""

    Nose = 30
    Chin = 8
    LeftEye = 36
    RightEye = 45  # good: 45
    LeftMouth = 48
    RightMouth = 54  # 65
    image_points = np.array([
        (shape.part(Nose).x, shape.part(Nose).y),  # Nose tip (<- Nose: 30~35)
        (shape.part(Chin).x, shape.part(Chin).y),  # Chin   (<- Jaw: 0~16)
        (shape.part(LeftEye).x, shape.part(LeftEye).y),  # Left eye left corner (<- Left ey: 36~41)
        (shape.part(RightEye).x, shape.part(RightEye).y),  # Right eye right corner (<- Right eye: 42~47)
        (shape.part(LeftMouth).x, shape.part(LeftMouth).y),  # Left mouth corner (<- Inner lip: 60~67, Outer lip: 48~59)
        (shape.part(RightMouth).x, shape.part(RightMouth).y)
        # Right mouth corner (<- Inner lip: 60~67, Outer lip: 48~59)
    ], dtype="double")


    # Camera internals
    size = frame.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)



    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)


    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    #cv2.line(frame, p1, p2, (255, 0, 0), 2)


    # Stabilize the nose using "stabilizer.py (Kalmann Filter)"
    stable_nose = []
    pose_np = np.array([p1,p2]).flatten()
    for value, ps_stb in zip(pose_np, nose_stabilizers):
        ps_stb.update([value])
        stable_nose.append(ps_stb.state[0])
    stable_nose = np.reshape(stable_nose, (-1, 2))
    p1 = (int(stable_nose[0][0]), int(stable_nose[0][1]))
    p2 = (int(stable_nose[1][0]), int(stable_nose[1][1]))
    #cv2.line(frame, p1, p2, (0, 255, 0), 2)


    return (p1, p2)   # nose_start, nose_end

    # Display image
    #cv2.imshow("Output", frame)


'''
{
drawPolyline(im, landmarks, 0, 16); // Jaw line
drawPolyline(im, landmarks, 17, 21); // Left eyebrow
drawPolyline(im, landmarks, 22, 26); // Right eyebrow
drawPolyline(im, landmarks, 27, 30); // Nose bridge
drawPolyline(im, landmarks, 30, 35, true); // Lower nose
drawPolyline(im, landmarks, 36, 41, true); // Left eye
drawPolyline(im, landmarks, 42, 47, true); // Right eye
drawPolyline(im, landmarks, 48, 59, true); // Outer lip
drawPolyline(im, landmarks, 60, 67, true); // Inner lip
}
'''


#------------------------------------------
# Dlib: Load labels_id & face_descriptors of registered faces
(labels, fd_known) = load_registered_face()
#------------------------------------------




while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #win.clear_overlay()
    #win.set_image(frame)


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


        # ---------------------------------
        # Recognize by Dlib

        # Get the landmarks/parts for the face in box d.
        shape = sp(frame, d)

        # Compute the 128D vector that describes the face in img identified by
        # shape.  In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people. Here we just print
        # the vector to the screen.
        face_descriptor = facerec.compute_face_descriptor(frame, shape)

        fd_th = 0.5
        print(labels)
        #print(len(fd_known))

        min_dist = fd_th
        selected_label = None

        for id in labels.keys():
            dist = np.subtract(face_descriptor, fd_known[id])
            dist = np.sqrt(np.dot(dist,dist))
            print("id: %2d, dist: %4.2f" % (id, dist))
            if(dist < fd_th and dist < min_dist):
                selected_label = labels[id]
                min_dist = dist
        if(selected_label != None):
            print(selected_label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            conf_str = "{0:3.1f}".format(min_dist)
            name = selected_label + ", [" + conf_str + "]"
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (d.left(), d.top()), font, 1, color, stroke, cv2.LINE_AA)


        # ---------------------------------
        # Get the landmarks/parts for the face in box d.
        shape = predictor(frame, d)
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

        # Find Head Pose using the face landmarks and Draw them on the screen.
        (p1, p2) = draw_landmark_headpose(frame, shape)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = dx*dx + dy*dy
        roi_ratio = (d.right() - d.left()) / frame.shape[0]
        dist_ratio = dist / (frame.shape[0]*frame.shape[0]) * ((d.right() - d.left()) / frame.shape[0])

        roi_ratio_th = 0.3
        dist_ratio_th = 0.03
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
