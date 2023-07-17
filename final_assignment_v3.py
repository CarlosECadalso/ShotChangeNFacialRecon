import cv2
import dlib
from PIL import Image, ImageOps

THRESHOLD = 9

frame_counter = 0


def export_video(frames, output_video_name):
    print("Processing video")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # video codec
    height, width, layers = frames[0].shape  # Set the video dimensions
    # Set where the new frames will go to
    video_output = cv2.VideoWriter(output_video_name, fourcc, 20.0, (width, height))

    # Write each frame into the video
    for frame in frames:
        video_output.write(frame)

    video_output.release()  # releasing the video generated
    cv2.destroyAllWindows()


def get_input_video(video_name):
    # Import the video as a list of frames
    vid = cv2.VideoCapture(video_name)

    return vid


def track_face(frame, faces, face_trackers, current_face_ID):
    # For every face in faces (which are translated to the coordinates and size of the face)
    for (_x, _y, _w, _h) in faces:
        x = int(_x)
        y = int(_y)
        w = int(_w)
        h = int(_h)

        # Calculate the center point
        x_center = x + 0.5 * w
        y_center = y + 0.5 * h
        # This will store the face id
        matchedFid = None

        # For every face (gotten from the face ids in the face_trackers dict)
        for fid in face_trackers.keys():
            position_of_face = face_trackers[fid].get_position()

            tracted_x = int(position_of_face.left())
            tracted_y = int(position_of_face.top())
            tracted_w = int(position_of_face.width())
            tracted_h = int(position_of_face.height())

            # Calculate the centerpoint
            tracted_x_center = tracted_x + 0.5 * tracted_w
            tracted_y_center = tracted_y + 0.5 * tracted_h

            # This if statement provides a way to ensure that the face is
            # close enough to be the same face
            # If the new center is inside the previously detected frame of the face
            if ((tracted_x <= x_center <= (tracted_x + tracted_w)) and 
                (tracted_y <= y_center <= (tracted_y + tracted_h)) and 
                # If the previous center is inside the newly detected frame of the face
                (x <= tracted_x_center <= (x + w)) and 
                (y <= tracted_y_center <= (y + h))):
                # Since it is very likely that this is the new face, flip the
                # flag to say we got a match
                matchedFid = fid

        if matchedFid is None:
            # Create the face tracker for the unmatched face
            tracker = dlib.correlation_tracker()
            tracker.start_track(frame, dlib.rectangle(x-10, y-20, x+w+10, y+h+20))

            face_trackers[current_face_ID] = tracker

            # Increase the current_face_ID counter
            current_face_ID += 1
    
    # We return this to keep track of the faceIDs
    return current_face_ID


def delete_old_faces(frame, face_trackers):
    faces_to_remove = []
    something_was_removed = False

    for fid in face_trackers.keys():
        # Remove faces if they are too far
        position_of_face = face_trackers[fid].get_position()
        tracted_w = int(position_of_face.width())
        tracted_h = int(position_of_face.height())

        if tracted_w < 30 or tracted_h < 30:
            faces_to_remove.append(fid)

        # If the quality of tracking is under an arbitrary a threshold
        # the fid is marked for deletion
        trackingQuality = face_trackers[fid].update(frame)
        if trackingQuality < THRESHOLD:
            faces_to_remove.append(fid)

    # Remove the faces marked for deletion
    for fid in faces_to_remove:
        something_was_removed = True
        face_trackers.pop(fid, None)

    return something_was_removed


def draw_rectangles_on_faces(frame, face_trackers, actor_names):
    for fid in face_trackers.keys():
        # Get the position of the face
        tracked_position = face_trackers[fid].get_position()
        x = int(tracked_position.left())
        y = int(tracked_position.top())
        w = int(tracked_position.width())
        h = int(tracked_position.height())

        if (x < frame.shape[0] and y < frame.shape[1]) and (x + w < frame.shape[0] and y + h < frame.shape[1]):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # If the actor was found among the pictures, also print the name
            if fid in actor_names.keys():
                cv2.putText(frame, fid, (int(x + w/2), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Unidentified Actor", (int(x + w/2), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def face_detection(input_video_name):
    # Get the video
    vid = get_input_video(input_video_name)

    # Create a CascadeClassifier, an object for face detection using Haar 
    # feature-based cascade classifiers as proposed by Paul Viola and Michael Jones
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # A dictionary of faces with ids as keys
    face_trackers = dict()

    # A dictionary with the names of actors using the same keys as face_trackers
    actor_names = dict()

    # A list of frames that will later be compiled into a video
    list_of_frames = list()

    # Get the first frame of the video
    # This first time is outside the loop to emulate a "do{} while()" loop
    frame_exists, frame = vid.read()

    # This is a temporary id for identifying faces
    current_face_ID = 0

    counter = 0

    # Loop through each frame in the video
    while frame_exists:
        # Convert each frame to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # equalized = clahe.apply(grayscale_frame)

        # Detect faces in the frame using the previosly defined CascadeClassifier
        # Note: Faces are detected in grayscale, but the ractagle is drawn over
        # the coloured frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Track faces
        # if counter > 9:
        #     current_face_ID = track_face(frame, faces, face_trackers, current_face_ID)
        #     counter += 1

        current_face_ID = track_face(frame, faces, face_trackers, current_face_ID)
        
        # This cleans up old faces that are no longer visible
        delete_old_faces(frame, face_trackers)

        # Draw the rectangle on the image
        new_frame = draw_rectangles_on_faces(frame, face_trackers, actor_names)

        # Add the frame to the list
        list_of_frames.append(new_frame)

        # Get the following frame for the next loop iteration
        frame_exists, frame = vid.read()
        
    # Export the frames into a new video
    export_video(list_of_frames, "output_" + input_video_name)


face_detection("shawshank_cut.mp4")
