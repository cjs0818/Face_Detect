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

    size = frame.shape

    # Camera internals
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

    cv2.line(frame, p1, p2, (255, 0, 0), 2)

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

        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Find Head Pose using the face landmarks and Draw them on the screen.
        draw_landmark_headpose(frame, shape)


        # Draw ROI box
        color = (255, 0, 0)  # BGR 0-255
        x = d.left()
        y = d.top()
        end_cord_x = d.right()
        end_cord_y = d.bottom()
        stroke = 2
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)



    # Display the resulting frame
    cv2.imshow('frame',frame)

    #win.add_overlay(dets)


    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
