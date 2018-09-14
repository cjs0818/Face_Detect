# -*-coding: utf-8 -*-

import cv2


"""Draws squares around detected faces in the given image."""

import argparse

# [START vision_face_detection_tutorial_imports]
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw
# [END vision_face_detection_tutorial_imports]


def highlight_faces(image, faces, output_filename):
    """Draws a polygon around the faces, then saves to output_filename.

    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    """
    im = Image.open(image)
    draw = ImageDraw.Draw(im)

    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')

    im.save(output_filename)

def detect_face(image_file):
    """Detects faces in an image."""
    client = vision.ImageAnnotatorClient()

    #with io.open(path, 'rb') as image_file:
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


def detect_faces_image(input_filename):
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


def detect_faces_camera(path):
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


#----------------------------------------------------
# 메인 함수
#----------------------------------------------------
if __name__ == '__main__':
	detect_faces_image(''./data/R1024x0.jpg")
