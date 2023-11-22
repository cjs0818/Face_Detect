# Face Detection, Head Pose Detection and Face Recognition with OpenCV(python)

## Install Python, pyenv, virtualenv
- Install python3   (v3.6.5 is recommended, since v3.7 & above is not compatible with tensorflow yet)
	- Ref: [https://beomi.github.io/2016/12/28/HowToSetup-Virtualenv-VirtualenvWrapper/](https://beomi.github.io/2016/12/28/HowToSetup-Virtualenv-VirtualenvWrapper/)
	```
	$ brew install python3
	```
- Install pyenv (to manage several versions of python in one PC)
	- Ref: [https://jiyeonseo.github.io/2016/07/27/install-pyenv/](https://jiyeonseo.github.io/2016/07/27/install-pyenv/)

	- In macOS,
	```
	$ brew install pyenv
	$ brew upgrade pyenv  # It may generate some error message like "pyenv 1.2.7 already installed", but you can neglect it.
	```
    - In Linux,
    ```
    $ sudo apt-get install pyenv
    ```
    - PATH 설정 (~/.bash_profile 또는 ~/.bashrc에 아래 내용 추가)
    ```
    export PYENV_ROOT="$HOME/.pyenv"
	export PATH="$PYENV_ROOT/bin:$PATH"
	eval "$(pyenv init -)"
	source ~/.bash_profile
    ```
    - 설치할 수 있는 목록 확인
    ```
    $ pyenv install --list
    ```
    - 원하는 python 버전 설치 준비
    ```
    $ pyenv install 3.6.5
    ```
    - 설치 준비된 python 버전 확인
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


## Download git source files & create virtualenv
- Download git source files
```
$ git clone https://github.com/cjs0818/Face_Detect.git
$ cd Face_Detect
```

- Note, **tensorflow is not compatible with python3.7**.
  - Then, you should use python 3.6.5 (it can be easily switched using pyenv which is already described above)
  ```
  $ pyenv shell 3.6.5
  ```
  - You need to install the fixed custom builds of tensorflow whose version is fit to your system from [https://github.com/lakshayg/tensorflow-build](https://github.com/lakshayg/tensorflow-build)
  ```
  $ pip3 install --ignore-installed --upgrade /path/to/binary.whl
  ```
  For my system, (python 3.6.5 and tensorflow 1.9.0)
  ```
  $ pip3 install --ignore-installed --upgrade tensorflow-1.9.0-cp36-cp36m-macosx_10_13_x86_64.whl
  ```

- If you do not have **Cmake** installed, install it first to use **Dlib** module.
    - Download the compressed Cmake software from [https://cmake.org/download/](https://cmake.org/download/)
      - [https://cmake.org/files/v3.12/cmake-3.12.2.tar.gz](https://cmake.org/files/v3.12/cmake-3.12.2.tar.gz)
    - Uncompress the downloaded file
    - Move inside the uncompressed folder (cmake-3.12.2)
    - Install Cmake as follows (**Note: It takes some time!**)
  ```
  $ ./bootstrap && make && make install
  ```

- Create virtualenv
```
$ virtualenv -p python3 venv   # <-- virtual env -p <python version> <가상 environment name
```
```
 $ virtualenv -p python3 venv   # Make sure you are using python 3.6.5
```
  - If you have already had venv with different version of python3 (i.e, python3.7.0), then you first need to remove the virtualenv folder (venv) and recreate a new venv with python3.6.5 



- Go into the virtualenv
```
$ source venv/bin/activate
```
	- When you exit the virtualenv, just type `deactivate` in the environment.
	```
    $ (venv) deactivate
    ```
- Install the required modules which are written in `requirements.txt.` file. (**It takes some time!!!**)
```
$ pip3 install -r requirements.txt
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

## Install ffmpeg
### For the use of AudioSegment.from_mp3()
  - Install using conda
    ```
    conda install ffmpeg
    ```

 or
 
 - Download the source file into '~/work' folder
   ```
   git clone https://git.ffmpeg.org/ffmpeg.git
   ```
- Then, configure, compile and install
  ```
  ./configure
  make
  make install
  ```

## Install large model data
### For Head Pose Detection (Dlib)
 - Download a trained facial shape predictor from:
 	- [http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2](http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2)
### For Face Recognition (Dlib)
 - Da trained facial shape predictor and recognition model from:
 	- [http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2)
 	- [http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2/)
