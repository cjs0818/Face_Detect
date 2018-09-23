# -*- coding: utf-8 -*-


# ----------------------------
#   Action Event Detection: action_detection/action_detection.py
# ----------------------------


ACTION_STATE_IDLE = 0
ACTION_EVENT_APPROACH = 1
ACTION_EVENT_DISAPPEAR = 2
ACTION_STATE_FACE_DETECTED = 3


class Event_Detector():
    def __init__(self):
        self.event = ACTION_STATE_IDLE
        self.state = ACTION_STATE_IDLE
        self.state_prev = ACTION_STATE_IDLE
        self.approach_cnt = 0
        self.approach_cnt_th = 5
        self.disappear_cnt = 0
        self.disappear_cnt_th = 5
        self.event_label = []

        self.id = -1

    def reset(self):
        self.event = ACTION_STATE_IDLE
        self.state = ACTION_STATE_IDLE
        self.state_prev = ACTION_STATE_IDLE
        self.approach_cnt = 0
        self.approach_cnt_th = 5
        self.disappear_cnt = 0
        self.disappear_cnt_th = 5
        self.event_label = []


    def approach_disappear(self, fr_labels, fr_box, min_width_id):

        self.event = ACTION_STATE_IDLE

        if self.state == ACTION_STATE_IDLE:
            self.disappear_cnt = 0
            if len(fr_labels) > 0:
                self.approach_cnt += 1
                if self.approach_cnt >= self.approach_cnt_th:
                    self.state = ACTION_STATE_FACE_DETECTED
                    self.event = ACTION_EVENT_APPROACH
                    print("! --------  APPROACH: {}  -------".format(fr_labels[min_width_id]))
            else:
                self.approach_cnt = 0
        elif self.state == ACTION_STATE_FACE_DETECTED:
            self.approach_cnt = 0
            if len(fr_labels) > 0:
                self.disappear_cnt = 0
            else:
                self.disappear_cnt += 1
                if self.disappear_cnt >= self.disappear_cnt_th:
                    self.state = ACTION_STATE_IDLE
                    self.event = ACTION_EVENT_DISAPPEAR
                    print("!        --------  DISAPPEAR  -------")





        #print("   Event State = %1d" % self.state)

        return (self.state, self.event)

    def get_state(self):
        return self.state
    def put_state(self, state):
        self.state = state
