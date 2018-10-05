# app.py
import requests

import json



def main():

    text = 'Hi, Hello!'
    user_key = 'DeepTasK'

    data_send = {
        'lang': 'ko',
        'query': text,
        'sessionId': user_key,
        'timezone': 'Asia/Seoul'
    }

    data_header = {
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': 'Bearer 9d10041bdb9c4c68a88b7899ca1540c1'  # Dialogflow의 Client access token 입력
    }

    url = 'http://localhost:60000/message'


    res = requests.post(url,
                        data=json.dumps(data_send),
                        headers=data_header)


if __name__ == '__main__':
    main()
