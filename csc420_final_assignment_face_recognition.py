import pickle

import cv2
import face_recognition

THRESHOLD = 9

FACE_POSITION_THRESHOLD = 0.9

FACE_SIZE_THRESHOLD = 0.2

SCD_THRESHOLD = 1000000

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


def track_and_identify_faces(frame, prev_frame, is_different_scene, face_cascade, eye_cascade,
                             face_trackers, actor_names, current_face_ID):
    # Convert each frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # equalized = clahe.apply(grayscale_frame)
    confirmed_faces = dict()

    # Detect faces in the frame using the previosly defined CascadeClassifier
    # Note: Faces are detected in grayscale, but the ractagle is drawn over
    # the coloured frame
    faces = face_cascade.detectMultiScale(grayscale_frame, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    for (nx, ny, nw, nh) in faces:
        is_tracked = False  # A flag that checks whether the face was already seen in the previous frame
        if prev_frame is not None:  # and not is_different_scene
            for face_id in face_trackers.keys():
                x = face_trackers[face_id][0]
                y = face_trackers[face_id][1]
                w = face_trackers[face_id][2]
                h = face_trackers[face_id][3]

                # If the face is close enough in location and size to the previous face then it is likely the same
                # face
                if (abs(x - nx) < frame.shape[0] * FACE_POSITION_THRESHOLD) and \
                        (abs(y - ny) < frame.shape[1] * FACE_POSITION_THRESHOLD) and \
                        (w * (1 - FACE_SIZE_THRESHOLD) < nw < w * (1 + FACE_SIZE_THRESHOLD)) and \
                        (h * (1 - FACE_SIZE_THRESHOLD) < nh < h * (1 + FACE_SIZE_THRESHOLD)) and \
                        not is_different_scene:
                    # If the box is within the frame
                    if (x < frame.shape[0] and y < frame.shape[1]) and (
                            x + w < frame.shape[0] and y + h < frame.shape[1]):
                        # Detect eyes
                        face_gray = grayscale_frame[ny:ny + nh, nx:nx + nw]
                        cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
                        eyes = eye_cascade.detectMultiScale(face_gray)
                        # for (ex, ey, ew, eh) in eyes:
                        #     cv2.rectangle(frame, (nx + ex, ny + ey), (nx + ex + ew, ny + ey + eh), (255, 0, 0), 2)
                        if len(eyes) > 0:
                            # Draw rectangle around the face
                            cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (255, 255, 255), 2)
                            confirmed_faces[face_id] = (nx, ny, nw, nh)
                    is_tracked = True  # The face has been tracked
        # In case this is a new face
        if not is_tracked:
            face_trackers[current_face_ID] = (nx, ny, nw, nh)
            # if (nx < frame.shape[0] and ny < frame.shape[1]) and (
            #         nx + nw < frame.shape[0] and ny + nh < frame.shape[1]):
            #     cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
            #     pass
            current_face_ID = current_face_ID + 1
    return current_face_ID, confirmed_faces


def process_video(input_video_name):
    # Import the data encoding
    data_encodings = pickle.loads(open("./encodings.txt", "rb").read())

    frame_num = 0
    frames_since_transition = 20

    # Get the video
    vid = get_input_video(input_video_name)

    # Create a CascadeClassifier, an object for face detection using Haar 
    # feature-based cascade classifiers as proposed by Paul Viola and Michael Jones
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # A dictionary of faces with ids as keys
    face_trackers = dict()

    # A dictionary with the names of actors using the same keys as face_trackers
    actor_names = dict()

    # A list of frames that will later be compiled into a video
    list_of_frames = list()

    # Get the first frame of the video
    # This first time is outside the loop to emulate a "do{} while()" loop
    frame_exists, frame = vid.read()
    frame_num = frame_num + 1

    # This will store the previous frame
    prev_frame = frame.copy()

    # This is a temporary id for identifying faces
    current_face_ID = 0

    # Making a histogram for scene change algorithm
    hist_size = 256
    hist_range = [0, 256, 0, 256, 0, 256]
    prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None, [hist_size, hist_size, hist_size], hist_range)

    # Get the image for marking scene changes
    scene_change_frame = cv2.imread('./vid/scene_change.png')

    # This crops the scene_change_frame to be the same size as the frame
    width = frame.shape[1]
    height = frame.shape[0]
    dim = (width, height)
    scene_change_frame = cv2.resize(scene_change_frame, dim)

    max_face_id = 0
    recognized_id = 0

    # Loop through each frame in the video
    while frame_exists:
        # A flag to indicate when the scene changes
        is_different_scene = False

        # Calculate histogram for current frame
        curr_hist = cv2.calcHist([frame], [0, 1, 2], None, [hist_size, hist_size, hist_size], hist_range)

        # Calculate histogram difference between current and previous frames
        diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)

        # If histogram difference is above threshold, scene change detected
        if diff > SCD_THRESHOLD and frames_since_transition >= 30:
            frames_since_transition = 0
            print('Scene change detected at ' + str(frame_num))
            print(str(diff))
            attach_frame = scene_change_frame.copy()
            cv2.putText(attach_frame, str(frame_num), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            for i in range(10):
                list_of_frames.append(attach_frame)
            is_different_scene = False
            # print(diff)
        else:
            is_different_scene = False
            frames_since_transition = frames_since_transition + 1

        # Update previous histogram
        prev_hist = curr_hist

        # Note: The following uses the face_recognition module. This module also includes face detection.
        # However, I implemented face detection using cv2 for the sake of relying less on already existing
        # modules while doing the assignment. I utilize this module for face recognition because I did not
        # find any other reliable way of doing it.
        # Change the format of the frame into RGB to face_recognition to work with it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        previous_boxes = dict()
        named_boxes = dict()

        # Detect faces (again)
        boxes = face_recognition.face_locations(frame_rgb, model="hog")

        for (by1, bx2, by2, bx1) in boxes:  # Boxes
            for key in previous_boxes.keys():  # Previously seen boxes
                (sby1, sbx2, sby2, sbx1) = previous_boxes[key]
                # If the face is close enough in location and size to the previous face then it is likely the same
                # face, and it is identified, then skip the identification process and simply draw the rectangle
                if (abs(sbx1 - bx1) < frame.shape[0] * FACE_POSITION_THRESHOLD) and \
                        (abs(sby1 - by1) < frame.shape[1] * FACE_POSITION_THRESHOLD) and \
                        (abs(sbx2 - bx2) < frame.shape[0] * FACE_POSITION_THRESHOLD) and \
                        (abs(sby2 - by2) < frame.shape[1] * FACE_POSITION_THRESHOLD) and \
                        not is_different_scene and \
                        (sbx1 < frame.shape[0] and sby1 < frame.shape[1]) and \
                        (sbx2 < frame.shape[0] and sby2 < frame.shape[1]) and \
                        named_boxes[key] != "Unidentified Actor":
                    boxes.remove((by1, bx2, by2, bx1))
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    if by1 - 15 > 15:
                        text_y = by1 - 15
                    else:
                        text_y = by1 + 15
                    cv2.putText(frame, actor_name, (bx1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    break
                # else:
                #     previous_boxes[max_face_id] = (by1, bx2, by2, bx1)
                #     recognized_id += 1
                #     named_boxes[max_face_id] = "Unidentified Actor"

        # A list of names to be used later
        recognized_faces_names = list()

        # Default name in case no face is found identifiable
        resulting_name = "Unidentified Actor"

        # Compute the facial embeddings for each face to differentiate them (and later associate them with
        # the actor name)
        encodings = face_recognition.face_encodings(frame_rgb, boxes)

        # For each facial embeddings
        for encoding in encodings:
            # Get the faces that match
            matches = face_recognition.compare_faces(data_encodings["encodings"], encoding)
            # If there is a match
            if True in matches:
                matched_indexes = []
                # Go through each true match
                for i, result in enumerate(matches):
                    if result:
                        # Attach the associated index to matched indexes to mark it as a possible recognition
                        matched_indexes.append(i)
                # This dictionary and the subsequent loop is meant to count how many votes the match has
                counts = dict()
                for i in matched_indexes:
                    resulting_name = data_encodings["names"][i]
                    counts[resulting_name] = counts.get(resulting_name, 0) + 1
                # Then take the match with the highest amount of votes as the correct result.
                # Do note that if there are no matches, the resulting_name will be "Unidentified Actor"
                # if max(counts.values()) < 10:
                resulting_name = max(counts, key=counts.get)
                # else:
                #     resulting_name = "Unidentified Actor"
        recognized_faces_names.append(resulting_name)

        for ((y1, x2, y2, x1), actor_name) in zip(boxes, recognized_faces_names):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            previous_boxes[max_face_id] = (y1, x2, y2, x1)
            named_boxes[max_face_id] = resulting_name
            max_face_id += 1
            if y1 - 15 > 15:
                text_y = y1 - 15
            else:
                text_y = y1 + 15
            cv2.putText(frame, actor_name, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # See function above
        current_face_ID, faces_in_frame = track_and_identify_faces(frame, prev_frame, is_different_scene,
                                                                   face_cascade, eye_cascade, face_trackers,
                                                                   actor_names, current_face_ID)

        # Add the frame to the list
        list_of_frames.append(frame)

        # Get the following frame for the next loop iteration
        prev_frame = frame.copy()
        frame_exists, frame = vid.read()
        frame_num = frame_num + 1
        print(frame_num)

    # Export the frames into a new video
    export_video(list_of_frames, "./vid/out/output_" + VIDEO_TO_RUN + "_cut.mp4")


VIDEO_TO_RUN = "presentation"

process_video("./vid/cut/" + VIDEO_TO_RUN + "_cut.mp4")

# # A flag to indicate when the scene changes
# is_different_scene = False
#
# # Convert each frame to grayscale
# grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# # equalized = clahe.apply(grayscale_frame)
#
# # Detect faces in the frame using the previosly defined CascadeClassifier
# # Note: Faces are detected in grayscale, but the ractagle is drawn over
# # the coloured frame
# faces = face_cascade.detectMultiScale(grayscale_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
# for (nx, ny, nw, nh) in faces:
#     is_tracked = False  # A flag that checks whether the face was already seen in the previous frame
#     if prev_frame is not None and not is_different_scene:
#         for face_id in face_trackers.keys():
#             x = face_trackers[face_id][0]
#             y = face_trackers[face_id][1]
#             w = face_trackers[face_id][2]
#             h = face_trackers[face_id][3]
#
#             # If the face is close enough in location and size to the previous face then it is likely the same
#             # face
#             if (abs(x - nx) < frame.shape[0] * FACE_POSITION_THRESHOLD) and \
#                     (abs(y - ny) < frame.shape[1] * FACE_POSITION_THRESHOLD) and \
#                     (w * (1 - FACE_SIZE_THRESHOLD) < nw < w * (1 + FACE_SIZE_THRESHOLD)) and \
#                     (h * (1 - FACE_SIZE_THRESHOLD) < nh < h * (1 + FACE_SIZE_THRESHOLD)) and \
#                     not is_different_scene:
#                 # If the box is within the frame
#                 if (x < frame.shape[0] and y < frame.shape[1]) and (
#                         x + w < frame.shape[0] and y + h < frame.shape[1]):
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     # If the actor was found among the pictures, also print the name
#                     if face_id in actor_names.keys():
#                         cv2.putText(frame, face_id, (int(x + w / 2), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                     (255, 255, 255), 2)
#                     else:
#                         cv2.putText(frame, "Unidentified Actor", (int(x + w / 2), y),
#                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#                 is_tracked = True  # The face has been tracked
#     # In case this is a new face
#     if not is_tracked:
#         face_trackers[current_face_ID] = (nx, ny, nw, nh)
#         if (nx < frame.shape[0] and ny < frame.shape[1]) and (
#                 nx + nw < frame.shape[0] and ny + nh < frame.shape[1]):
#             cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)

# Track faces
# if counter > 9:
#     current_face_ID = track_face(frame, faces, face_trackers, current_face_ID)
#     counter += 1

# current_face_ID = track_face(frame, faces, face_trackers, current_face_ID)

# This cleans up old faces that are no longer visible
# delete_old_faces(frame, face_trackers)
#
# # Draw the rectangle on the image
# new_frame = draw_rectangles_on_faces(frame, face_trackers, actor_names)

# def track_face(frame, faces, face_trackers, current_face_ID):
#     # For every face in faces (which are translated to the coordinates and size of the face)
#     for (_x, _y, _w, _h) in faces:
#         x = int(_x)
#         y = int(_y)
#         w = int(_w)
#         h = int(_h)
#
#         # Calculate the center point
#         x_center = x + 0.5 * w
#         y_center = y + 0.5 * h
#         # This will store the face id
#         matchedFid = None
#
#         # For every face (gotten from the face ids in the face_trackers dict)
#         for fid in face_trackers.keys():
#             position_of_face = face_trackers[fid].get_position()
#
#             tracted_x = int(position_of_face.left())
#             tracted_y = int(position_of_face.top())
#             tracted_w = int(position_of_face.width())
#             tracted_h = int(position_of_face.height())
#
#             # Calculate the centerpoint
#             tracted_x_center = tracted_x + 0.5 * tracted_w
#             tracted_y_center = tracted_y + 0.5 * tracted_h
#
#             # This if statement provides a way to ensure that the face is
#             # close enough to be the same face
#             # If the new center is inside the previously detected frame of the face
#             if ((tracted_x <= x_center <= (tracted_x + tracted_w)) and
#                     (tracted_y <= y_center <= (tracted_y + tracted_h)) and
#                     # If the previous center is inside the newly detected frame of the face
#                     (x <= tracted_x_center <= (x + w)) and
#                     (y <= tracted_y_center <= (y + h))):
#                 # Since it is very likely that this is the new face, flip the
#                 # flag to say we got a match
#                 matchedFid = fid
#
#         if matchedFid is None:
#             # Create the face tracker for the unmatched face
#             tracker = dlib.correlation_tracker()
#             tracker.start_track(frame, dlib.rectangle(x - 10, y - 20, x + w + 10, y + h + 20))
#
#             face_trackers[current_face_ID] = tracker
#
#             # Increase the current_face_ID counter
#             current_face_ID += 1
#
#     # We return this to keep track of the faceIDs
#     return current_face_ID
#
#
# def delete_old_faces(frame, face_trackers):
#     faces_to_remove = []
#     something_was_removed = False
#
#     for fid in face_trackers.keys():
#         # Remove faces if they are too far
#         position_of_face = face_trackers[fid].get_position()
#         tracted_w = int(position_of_face.width())
#         tracted_h = int(position_of_face.height())
#
#         if tracted_w < 30 or tracted_h < 30:
#             faces_to_remove.append(fid)
#
#         # If the quality of tracking is under an arbitrary a threshold
#         # the fid is marked for deletion
#         trackingQuality = face_trackers[fid].update(frame)
#         if trackingQuality < THRESHOLD:
#             faces_to_remove.append(fid)
#
#     # Remove the faces marked for deletion
#     for fid in faces_to_remove:
#         something_was_removed = True
#         face_trackers.pop(fid, None)
#
#     return something_was_removed
#
#
# def draw_rectangles_on_faces(frame, face_trackers, actor_names):
#     for fid in face_trackers.keys():
#         # Get the position of the face
#         tracked_position = face_trackers[fid].get_position()
#         x = int(tracked_position.left())
#         y = int(tracked_position.top())
#         w = int(tracked_position.width())
#         h = int(tracked_position.height())
#
#         if (x < frame.shape[0] and y < frame.shape[1]) and (x + w < frame.shape[0] and y + h < frame.shape[1]):
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#             # If the actor was found among the pictures, also print the name
#             if fid in actor_names.keys():
#                 cv2.putText(frame, fid, (int(x + w / 2), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#             else:
#                 cv2.putText(frame, "Unidentified Actor", (int(x + w / 2), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                             (255, 255, 255), 2)
#
#     return frame
