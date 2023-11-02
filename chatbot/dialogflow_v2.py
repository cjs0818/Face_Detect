# -*- coding: utf-8 -*-

#----------------------------------------
# Web API for Danbee.ai ChatBot Platform
#----------------------------------------

import requests
import json
import csv

from google.cloud import dialogflow
import sys
import os



class ChatBot():
    def __init__(self, project_id=0, session_id=0, language_code='ko-KR'):
        self.project_id = project_id
        self.session_id = session_id
        self.language_code = language_code

        # You first should include this command to set GOOGLE_APPLICATION_CREDENTIALS and PYTHONPATH
        if sys.platform == "linux" or sys.platform == "linux2":
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/jschoi/work/Face_Detect/chatbot/receptionbot-3b113-b2ed90c08841.json'
            os.environ["PYTHONPATH"] = '/home/jschoi/work/sHRI_base:$PYTHONPATH'
        elif sys.platform == "darwin":
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/jschoi/work/Face_Detect/chatbot/receptionbot-3b113-b2ed90c08841.json'
            os.environ["PYTHONPATH"] = '/Users/jschoi/work/sHRI_base:$PYTHONPATH'


    def print_kor(self, text):
        #print(json.dumps(text, indent=4, ensure_ascii=False))
        print(text)


    def event_api(self, event, user_key):
        data_send = {
            'chatbot_id': 'c54e4466-d26d-4966-af1f-ca4d087d0c4a',
            'parameters': event
        }
        data_header = {
            "Content-Type": "application/json;charset=UTF-8"
        }
        event_url = "https://danbee.ai/chatflow/54ae138f-fa8f-404f-8975-8ac6a3c45c35/eventFlow.do"
        res = requests.post(event_url,
                            data=json.dumps(data_send),
                            headers=data_header)
        data_receive = res.json()

        message = data_receive['responseSet']['result']['result'][0]['message']  # <- danbee json 포맷 분석 결과
        print("----- Event API ------")
        print(message)
        print("  ")

        return data_receive

    def event_api_dialogflow(self, event_name, event_data, user_key):
        # --------------------------------
        # Dialogflow에 요청
        # --------------------------------
        '''
        data_send = {
            'lang': 'ko',
            'event': {
                'name': '<Event_name>',
                'data': {
                    '<parameter_name1>': '<parameter_value1>',
                    ...
                }
            },
            'sessionId': user_key,
            'timezone': 'Asia/Seoul'
        }
        '''
        data_send = {
            'lang': 'ko',
            'event': {
                'name': event_name,
                'data': event_data
            },
            'sessionId': user_key,
            'timezone': 'Asia/Seoul'
        }


        data_header = {
            'Content-Type': 'application/json; charset=utf-8',
            #'Authorization': 'Bearer 9d10041bdb9c4c68a88b7899ca1540c1'  # Dialogflow의 Client access token 입력 (/query, /contexts, /userEntities에 대해서)
             'Authorization': 'Bearer 73d41dcf34e849a1b8c911559a790112'  # Dialogflow의 Client access token 입력 (/query, /contexts, /userEntities에 대해서)
        }
        # Caution: The developer access token is used for /intents and /entities endpoints. The client access token is used for /query, /contexts, and /userEntities endpoints.

        dialogflow_url = 'https://api.dialogflow.com/v1/query?v=20150910'


        res = requests.post(dialogflow_url,
                            data=json.dumps(data_send),
                            headers=data_header)

        # --------------------------------
        # 대답 처리
        # --------------------------------
        if res.status_code != requests.codes.ok:
            return ERROR_MESSAGE

        data_receive = res.json()

        #print(data_receive)

        message = data_receive['result']['fulfillment']['speech']


        #print(data_send)
        self.print_kor(user_key + ": " + "[Event] '" + event_name + "' with data ")
        self.print_kor(event_data)
        self.print_kor("      [receptionbot]: " + message)


        return data_receive

    # ----------------------------------------------------
    # Danbee.ai에서 대답 구함
    # ----------------------------------------------------
    def get_answer_danbee(self, text, user_key):
        # --------------------------------
        # Danbee 요청
        # --------------------------------
        data_send = {
            'chatbot_id': self.chatbot_id,      # 'c54e4466-d26d-4966-af1f-ca4d087d0c4a'
            'input_sentence': text
        }
        data_header = {
            "Content-Type": "application/json;charset=UTF-8"
        }
        danbee_chatflow_url = 'https://danbee.ai/chatflow/engine.do'

        res = requests.post(danbee_chatflow_url,
                            data=json.dumps(data_send),
                            headers=data_header)
        # --------------------------------
        # 대답 처리
        # --------------------------------
        '''
        if res.resultStatus != requests.codes.ok:
            return ERROR_MESSAGE
        '''

        data_receive = res.json()

        message = data_receive['responseSet']['result']['result'][0]['message']


        #answer = data_receive['result']

        print("\n")
        #print(user_key, ": ", text)
        #print("      [receptionbot]", ": ", message)

        sentence = user_key + ": " + text
        self.print_kor(sentence)
        sentence = "      [receptionbot]: " + message
        self.print_kor(sentence)


        self.print_kor(data_receive)

        return data_receive



    # ----------------------------------------------------
    # Dialogflow에서 대답 구함
    # ----------------------------------------------------
    def detect_intent(self, project_id, session_id, text, language_code):
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(project_id, session_id)

        text_input = dialogflow.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.QueryInput(text=text_input)
        response = session_client.detect_intent(request={"session": session, "query_input": query_input})

        return response.query_result.fulfillment_text

    '''
    def get_answer_dialogflow(self, text, user_key, context_flag=False, context_value=""):
        # --------------------------------
        # Dialogflow에 요청
        # --------------------------------
        data_send = {
            'lang': 'ko',
            'query': text,
            'sessionId': user_key,
            'timezone': 'Asia/Seoul'
        }
        if context_flag == True:
            data_send['contexts'] = context_value

        data_header = {
            'Content-Type': 'application/json; charset=utf-8',
            #'Authorization': 'Bearer 9d10041bdb9c4c68a88b7899ca1540c1'  # Dialogflow의 Client access token 입력
            'Authorization': 'Bearer 73d41dcf34e849a1b8c911559a790112'  # Dialogflow의 Client access token 입력 (/query, /contexts, /userEntities에 대해서)

        }

        dialogflow_url = 'https://api.dialogflow.com/v1/query?v=20150910'


        res = requests.post(dialogflow_url,
                            data=json.dumps(data_send),
                            headers=data_header)

        # --------------------------------
        # 대답 처리
        # --------------------------------
        if res.status_code != requests.codes.ok:
            print(data_send)
            return ERROR_MESSAGE

        data_receive = res.json()

        #print(data_receive)

        message = data_receive['result']['fulfillment']['speech']


        #print(user_key, ": ", text)
        #print("      [receptionbot]", ": ", message)
        self.print_kor(user_key + ": " + text)
        self.print_kor("      [receptionbot]: " + message)


        return data_receive
    '''

    # ----------------------------------------------------
    # POST test to heroku for database
    # ----------------------------------------------------
    def test_post(self, name):
        # --------------------------------
        # 요청
        # --------------------------------
        data_send = {
            'name': name
        }

        data_header = {
            'Content-Type': 'application/json; charset=utf-8'
        }

        dialogflow_url = 'https://heroku-dialogflow-chatbot.herokuapp.com/message_danbee'


        res = requests.post(dialogflow_url,
                            data=json.dumps(data_send),
                            headers=data_header)

        # --------------------------------
        # 대답 처리
        # --------------------------------
        if res.status_code != requests.codes.ok:
            return ERROR_MESSAGE

        data_receive = res.json()

        #print(data_receive)

        print('----------  Test from heroku -----------')
        print(json.dumps(data_receive, indent=4, ensure_ascii=False))
        print('----------------------------------------')


    # ----------------------------------------------------
    # database from CSV file
    # ----------------------------------------------------
    def get_datatbase(self, kind_of_guide):
        filename = 'RMI_researchers.csv'

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

        #json_data = json.dumps(dict, indent=4, ensure_ascii=False)
        #print(json_data)


        return dict


def detect_intent(project_id, session_id, text, language_code):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)

    text_input = dialogflow.TextInput(text=text, language_code=language_code)
    query_input = dialogflow.QueryInput(text=text_input)
    response = session_client.detect_intent(request={"session": session, "query_input": query_input})

    return response.query_result.fulfillment_text

def main():

    '''
    # --------------------------------
    # Start Chat with Danbee API
    # --------------------------------
    user_key = 'DeepTasK'
    chatbot_id = 'c54e4466-d26d-4966-af1f-ca4d087d0c4a'

    chat = Danbee(chatbot_id)

    content = '안녕하세요'
    res = chat.get_answer_danbee(content, user_key)
    message = res['responseSet']['result']['result'][0]['message']   # <- danbee json 포맷 분석 결과
    print(message)
    '''

    project_id = "receptionbot-3b113"
    session_id = "your-session-id"
    language_code = 'ko-KR'  # a BCP-47 language tag

    chat = ChatBot()

    #text = input("You: ")
    text = "안녕하세요"
    response = chat.detect_intent(project_id, session_id, text, language_code)
    print("Chatbot: " + response)


#----------------------------------------------------
# 메인 함수
#----------------------------------------------------
if __name__ == '__main__':

    main()
