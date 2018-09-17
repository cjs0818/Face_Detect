# Face Detection, Head Pose Detection and Face Recognition with OpenCV(python)

## Install Python, pyenv, virtualenv
- Install python3 
	- Ref: [https://beomi.github.io/2016/12/28/HowToSetup-Virtualenv-VirtualenvWrapper/](https://beomi.github.io/2016/12/28/HowToSetup-Virtualenv-VirtualenvWrapper/)
	```
	$ brew install python3
	```
- Install pyenv (to manage several versions of python in one PC)
	- Ref: [https://jiyeonseo.github.io/2016/07/27/install-pyenv/](https://jiyeonseo.github.io/2016/07/27/install-pyenv/)

	- In macOS,
	```
	$ brew install pyenv
	$ brew upgrade pyenv
	```
    - In Linux,
    ```
    $ sudo apt-get install pyenv
    ```
    - PATH 설정 (~/.bash_profile 또는 ~/.bashrc에 아래 내용 추가)
    ```
    $ export PYENV_ROOT="$HOME/.pyenv"
	$ export PATH="$PYENV_ROOT/bin:$PATH"
	$ eval "$(pyenv init -)"
	$ source ~/.bash_profile
    ```
    - 설치할 수 있는 목록 확인
    ```
    $ pyenv install --list
    ```
    - 원하는 python 버전 설치 준비
    ```
    $ pyenv install 3.6.5
    ```
    - 설치된 pyenv 확인
    ```
    $ pyenv versions
    ```
    - 원하는 python 버전으로 변환
    ```
    $ pyenv shell 3.6.5
    ```
    - 설치된 python 버전 확인
    ```
    $ python -V
    ```

- Install virtualenv
```
$ pip3 install virtualenv virtualenvwrapper
```
	- Virtualenv의 기본적 명령어
	```
    $ virtualenv -p python3 venv   # <- virtual env -p <python version> <가상 environment name
    ```

## Install OpenCV(python)

- Install opencv-python ([https://pypi.org/project/opencv-python/](https://pypi.org/project/opencv-python/))
```
$ pip3 install opencv-contrib-python # if you need both main and contrib modules (recommended)
$ pip3 install opencv-python  # if you need only main modules
```

- Reference for opencv stuffs: [https://opencv-python.readthedocs.io/en/latest/index.html](https://opencv-python.readthedocs.io/en/latest/index.html)

### Sample codes
- Camera Test: camera_test.py

```
# -*-coding: utf-8 -*-
import cv2

# cap 이 정상적으로 open이 되었는지 확인하기 위해서 cap.isOpen() 으로 확인가능
cap = cv2.VideoCapture(0)

# cap.get(prodId)/cap.set(propId, value)을 통해서 속성 변경이 가능.
# 3은 width, 4는 heigh

print('width: {0}, height: {1}'.format(cap.get(3),cap.get(4)))

cap.set(3,320)
cap.set(4,240)

while(True):
    # ret : frame capture결과(boolean)
    # frame : Capture한 frame
    ret, frame = cap.read()

    if (ret):
        # image를 Grayscale로 Convert함.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #cv2.imshow('frame', gray)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

## Install large model data
### For Head Pose Detection (Dlib)
 - Download a trained facial shape predictor from:
 	- [http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2](http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2)
### For Face Recognition (Dlib)
 - Da trained facial shape predictor and recognition model from:
 	- [http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2)
 	- [http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2/)