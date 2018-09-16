# ----------------------------------------------
# Face detection & Head Pose detection: by Dlib
# Face recognition: by OpenCV
# ----------------------------------------------


import numpy as np
import cv2
import pickle    # used for Face recognition by OpenCV

import sys
import dlib     # used for Face detectiob by Dlib
from skimage import io

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
nose_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1) for _ in range(6)]


# ----------------------------
# Face recognition: by OpenCV
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
#print(labels)



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


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''
    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
    	id_, conf = recognizer.predict(roi_gray)
    	if conf>=4 and conf <= 85:
    		#print(5: #id_)
    		#print(labels[id_])
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id_]
    		color = (255, 255, 255)
    		stroke = 2
    		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    '''


    #win.clear_overlay()
    #win.set_image(frame)


    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(frame, 1)
    print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))


        #------------------------------
        #  Recognize
        # print(x,y,w,h)
        roi_gray = gray[d.top():d.bottom(), d.left():d.right()]  # (ycord_start, ycord_end)
        roi_color = frame[d.top():d.bottom(), d.left():d.right()]

        #img_item = "7.png"
        #cv2.imwrite(img_item, roi_color)

        # recognize? deep learned model predict keras tensorflow pytorch scikit learn
        try:
            id_, conf = recognizer.predict(roi_gray)
            #if conf >= 4 and conf <= 85:
            if conf >= 45:
                # print(5: #id_)
                # print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                conf_str = "{0:3.1f}".format(conf)
                name = labels[id_] + ", [" + conf_str + "]"
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        except:
            pass
        # ...
        # ...
        #------------------------------


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
        print("roi_raatio: %3.2f, dist_ratio: %5.4f" % (roi_ratio, dist_ratio))
        if roi_ratio > roi_ratio_th and dist_ratio < dist_ratio_th:
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
        else:
            cv2.line(frame, p1, p2, (0, 255, 0), 2)


        # Draw ROI box
        color = (255, 0, 0)  # BGR 0-255
        x = d.left()
        y = d.top()
        end_cord_x = d.right()
        end_cord_y = d.bottom()
        stroke = 2
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)



        '''
        #-----------------------
        # head_pose_estimation
        #print(roi_color.shape)   # (214, 223, 3)
        #print((d.left(), d.top()), (d.right(), d.bottom()))
        #face_img = cv2.resize(roi_color, None, 128, 128, interpolation = cv2.INTER_CUBIC)
        face_img = cv2.resize(roi_color, (128, 128))   # <- (214, 223)
        marks = mark_detector.detect_marks(face_img)
        #print(marks.shape)

        # Convert the marks locations from local CNN to global image.
        marks *= (d.right() - d.left())
        marks[:, 0] += d.left()
        marks[:, 1] += d.top()

        # Uncomment following line to show raw marks.
        # mark_detector.draw_marks(
        #     frame, marks, color=(0, 255, 0))

        # Try pose estimation with 68 points.
        # pose = tuple of (pose[0], pose[1])
        #     pose[0] = rotation_vector,    # Rotation in axis-angle form
        #     pose[1] = translation_vector
        # example:
        #  pose = (array([[0.13766505],
        #        [-0.04562614],
        #        [-3.13804042]]),array([[-73.15168994],
        #                                [-44.24913723],
        #                                [-477.4216987]]))
        pose = pose_estimator.solve_pose_by_68_points(marks)

        print(pose)
        print("--------")

        # Stabilize the pose.
        stable_pose = []
        pose_np = np.array(pose).flatten()
        for value, ps_stb in zip(pose_np, pose_stabilizers):
            ps_stb.update([value])
            stable_pose.append(ps_stb.state[0])
        stable_pose = np.reshape(stable_pose, (-1, 3))
        print(stable_pose)

        # Uncomment following line to draw pose annotaion on frame.
        # pose_estimator.draw_annotation_box(
        #     frame, pose[0], pose[1], color=(255, 128, 128))

        # Uncomment following line to draw stable pose annotation on frame.
        pose_estimator.draw_annotation_box(
            frame, stable_pose[0], stable_pose[1], color=(128, 255, 128))
        '''


        '''
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        # Camera internals
        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        rotation_vector = stable_pose[0]
        translation_vector = stable_pose[1]
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        Nose = 30
        p1 = (int(marks[Nose][0]), int(marks[Nose][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (0, 255, 0), 2)
        '''
        #-----------------------




    # Display the resulting frame
    cv2.imshow('frame',frame)

    #win.add_overlay(dets)


    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
