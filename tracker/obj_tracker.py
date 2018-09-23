# -*- coding: utf-8 -*-

import dlib
import cv2

# ------------------------
# Object Tracking by Dlib correlation_tracker
#
#       obj_track = Obj_Tracker()
#       obj_track.start_tracking(frame, roi)  # Whenever you reset the initial roi for an object to track
#           obj_track.tracking(frame)
# ------------------------
class Obj_Tracker():
    def __init__(self):
        self.tracker = dlib.correlation_tracker()
        self.track_started = False
        self.track_running = False
        self.roi = []
        self.label = []
        self.min_dist = 0

    def start_tracking(self, image, roi):
        tracker = self.tracker
        rect = dlib.rectangle(roi)
        tracker.start_track(image, rect)
        self.track_started = True

    def tracking(self, image):
        self.track_running = True

        tracker = self.tracker
        tracker.update(image)
        new_roi = tracker.get_position()
        x  = int(new_roi.left())
        y  = int(new_roi.top())
        x1 = int(new_roi.right())
        y1 = int(new_roi.bottom())
        cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 255), 2)


        self.roi = dlib.rectangle(x,y, x1, y1)

        if x < 0 or y1 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
            #print("Lost!!!")
            self.track_started = False  # if the new_roi is out of the image, report the tracked object has been lost.
            self.track_running = False

        #print("obj_tracking!")

        #print("[left, top] = [%3d, %3d], [right, bottom] = [%3d, %3d]" %
        #      (int(new_roi.left()), int(new_roi.top()), int(new_roi.right()), int(new_roi.bottom())))


        return(self.roi)
