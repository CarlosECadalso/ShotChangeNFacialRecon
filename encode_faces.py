import numpy as np
import pickle
import cv2
import face_recognition
import os
from bing_image_downloader import downloader

THRESHOLD = 9
FACE_POSITION_THRESHOLD = 0.9
FACE_SIZE_THRESHOLD = 0.2
SCD_THRESHOLD = 1000000
frame_counter = 0
ACTOR_DATA_PATH = "./dataset/"
ORIGINAL_ACTOR_IMAGE_PATH = "./actor_data/"


def get_image_for_dataset(propt):
    # This function downloads 50 images from the internet to use in the dataset
    downloader.download(propt, limit=50, output_dir='./downloader_dataset/', adult_filter_off=False)


def get_actor_names():
    # Gets the names of the actors based on the directory
    name_dict = dict()
    name_file = open('./dataset/actor_names.txt', 'r')
    full_file_string = name_file.read()
    file_line_list = full_file_string.split('\n')
    for line in file_line_list:
        index, actor_name = line.split(':')
        name_dict[index] = actor_name
    name_file.close()
    return name_dict


def encode_faces():
    # This is a function that builds the dataset from the previously images downloaded
    imagePaths = list()
    # For each file in the folder
    for actor in os.listdir(ACTOR_DATA_PATH):
        # If the file is a dir (to ensure only the folders get selected) 
        if '.' not in actor:
            # For each image in the folder selected
            for image in os.listdir(ACTOR_DATA_PATH + actor):
                if image.endswith(".png") or image.endswith(".jpg"):
                    # Get the actor images into the imagePaths list
                    imagePaths.append(ACTOR_DATA_PATH + actor + "/" + image)

    # initialize the list of known encodings and known names
    knownEncodings = list()
    knownNames = list()
    actor_name_dict = get_actor_names()

    # For each image path
    for (i, imagePath) in enumerate(imagePaths):
        # Get the name of the actor
        index_to_find = imagePath.split("/")[-2]
        name_in_dict = actor_name_dict[index_to_find]

        # Read the image and identify the faces
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Store the names and encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name_in_dict)

    # Write the results into a file with the encodings
    data = {"encodings": knownEncodings, "names": knownNames}
    encoding_output = open("encodings.txt", "wb")
    encoding_output.write(pickle.dumps(data))
    encoding_output.close()