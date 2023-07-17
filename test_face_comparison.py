import cv2
import numpy as np

compare = list()

import os
import cv2
import numpy as np

ORIGINAL_ACTOR_IMAGE_PATH = "./actor_data/"


def identify_actor(face, face_cascade):
    curr_most_likely_actor = None
    curr_diff = 10000000
    for images in os.listdir(ORIGINAL_ACTOR_IMAGE_PATH):
        if images.endswith(".png") or images.endswith(".jpg"):
            actor_img = cv2.imread(ORIGINAL_ACTOR_IMAGE_PATH + images)

            # Convert the image to grayscale
            gray_actor = cv2.cvtColor(actor_img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            actor_face = face_cascade.detectMultiScale(gray_actor, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for detected_actor_face in actor_face:
                print(images, detected_actor_face)
                x, y, w, h = detected_actor_face
                actor_face_img = gray_actor.copy()[y:y + h, x:x + w]

                # norm_face = np.zeros(face.shape)
                # cv2.normalize(face, norm_face)
                # actor_face_norm = np.zeros(actor_face_img.shape)
                # cv2.normalize(actor_face_img, actor_face_norm)

                width = face.shape[1]
                height = face.shape[0]
                dim = (width, height)
                actor_face_img = cv2.resize(actor_face_img, dim)

                diff = cv2.absdiff(face, actor_face_img)
                confidence = diff.mean()

                if curr_diff > confidence:
                    curr_most_likely_actor = images
                    curr_diff = confidence
                    print("New Actor")
                    print(images, confidence)


for i in range(1):
    # Load the image
    image = cv2.imread("./actor_data/og/example.jpg")
    # image2 = cv2.imread("ralph_fiennes_" + str(i+1) + ".png")

    # Create a CascadeClassifier object for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces_dict = dict()

    # Draw rectangles around the detected faces
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the resulting image
        # faces_dict[i] = np.zeros((w, h, 3))
        faces_dict[i] = gray.copy()[y:y + h, x:x + w]
        identify_actor(faces_dict[i], face_cascade)
        # cv2_imshow(faces_dict[i])