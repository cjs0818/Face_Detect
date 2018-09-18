# -*- coding: utf-8 -*-

# ----------------------------------------------
# Face detection: by Dlib
# Camera Capture: by OpenCV
# ----------------------------------------------

import cv2
import dlib     # used for Face detectiob by Dlib

# ----------------------------
# Face detection: by Dlib
detector = dlib.get_frontal_face_detector()


cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)




while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


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



    # Display the resulting frame
    cv2.imshow('frame',frame)



    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
