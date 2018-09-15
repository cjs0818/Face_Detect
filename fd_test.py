# -*-coding: utf-8 -*-

import cv2
import io
import numpy as np


"""Draws squares around detected faces in the given image."""

import argparse

# [START vision_face_detection_tutorial_imports]
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw
# [END vision_face_detection_tutorial_imports]


def highlight_faces(image, faces):
    """Draws a polygon around the faces, then saves to output_filename.

    Args:
        image: a file containing the image with the faces.
        faces: a list of faces found in the file. This should be in the format returned by the Vision API.
        output_filename: the name of the image file to be created, where the
        faces have polygons drawn around them.
    """
    im = Image.open(image)
    draw = ImageDraw.Draw(im)

    for face in faces:
        box = [(vertex.x, vertex.y)
            for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')

    #im.save(output_filename)

def detect_face(image_file):
    """Detects faces in an image."""
    client = vision.ImageAnnotatorClient()

    #with io.open("./data/R1024x0.jpg", 'rb') as image_file:
    content = image_file.read()

    image = vision.types.Image(content=content)


    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('Faces:')

    for face in faces:
        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])

        print('face bounds: {}'.format(','.join(vertices)))

    return faces


def detect_faces_image(input_filename, output_filename):
    with open(input_filename, 'rb') as image:
        faces = detect_face(image)
        if len(faces) == 1:
            x = ''
        else:
            x = 's'
        print('Found {} face{}'.format(len(faces), x))

        print('Writing to file {}'.format(output_filename))
        # Reset the file pointer, so we can read the file again
        image.seek(0)
        highlight_faces(image, faces, output_filename)


def detect_faces_camera():
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



        ## Convert to an image, then write to a buffer.
        image_from_frame = Image.fromarray(np.uint8(frame))
        buffer = io.BytesIO()
        image_from_frame.save(buffer, format='PNG')
        buffer.seek(0)



        faces = detect_face(buffer)

        for face in faces:
            vertices = face.bounding_poly.vertices
            #print(vertices)

            x1 = vertices[0].x
            y1 = vertices[0].y
            x2 = vertices[2].x
            y2 = vertices[2].y
            print (x1,y1, x2,y2)
            #cv2.rectangle(buffer, (x1, y1), (x2, y2), (255, 0, 0), 2)



        if (ret):
            # image를 Grayscale로 Convert함.
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('frame', gray)

            #highlight_faces(buffer, faces)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



    cap.release()
    cv2.destroyAllWindows()


#----------------------------------------------------
# 메인 함수
#----------------------------------------------------
if __name__ == '__main__':
    #detect_faces_image("./data/R1024x0.jpg", "./data/out.jpg")
    detect_faces_camera()
