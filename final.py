import PySimpleGUI as sg
import os.path
import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle

now = datetime.now()

layout = [
    [sg.T(' ' * 25), sg.T(' ' * 25), sg.Image(r'E:/PycharmProjects/truminds/truminds (1).png')],
    [sg.T(' ' * 5),
     sg.Text('Face Counting', text_color='#5B6AAF', justification='center', size=(42, 1), font=("Helvetica, 25 bold"))],
    [sg.T(' ' * 5), sg.T(' ' * 5), sg.T(' ' * 5), sg.T(' ' * 5), sg.T(' ' * 5), sg.T(' ' * 5), sg.T(' ' * 5),
     sg.T(' ' * 5), sg.T(' ' * 5), #sg.Button('Upload Image', button_color=('white', '#5B6AAF'), size=(25, 3)),
     sg.Button('Start', button_color=('white', '#5B6AAF'), size=(25, 3)),
     sg.Button('Exit', button_color=('white', '#5B6AAF'), size=(25, 3))]
]

window = sg.Window('KUMAR SHUBHAM', default_element_size=(20, 1), grab_anywhere=False)

window.Layout(layout)

known_face_encodings = []
known_face_metadata = []
current_face_encoding = []
number_of_recognitions = 0
time_of_multiple_faces = None
time_of_no_faces = None
time_of_unknown_single_face = None
flag = True

def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("known faces backed up.")

def register_new_face(face_encoding, face_image):
    """
    Add a new person to our list of known faces
    """

    known_face_encodings.append(face_encoding)
    current_face_encoding.append(face_encoding)
    known_face_metadata.append({
        "first_seen": datetime.now(),
        "last_seen": datetime.now(),
        "face_image": face_image,
    })

def save_new_face_image(frame):
    """Register the face of candidate to be proctored"""
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    if(len(face_locations) !=  1) :
        print("Image not valid, Kindly select new image")
        return False
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    top, right, bottom, left = face_locations[0]
    face_image = small_frame[top:bottom, left:right]
    face_image = cv2.resize(face_image, (150, 150))
    register_new_face(face_encodings, face_image)
    return True

def lookup_known_face(face_encoding):
    """
    See if this is a face we already have in our face list
    """
    metadata = None
    global number_of_recognitions
    if len(known_face_encodings) == 0:
        return metadata

    #print(known_face_encodings[0])
    #print("Size of Known_face_encodings in lookup_known faces " + str(len(known_face_encodings)))
    #print("Size of curent_face_encodings in lookup_known faces " + str(len(current_face_encoding)))
    #print(str(type(known_face_encodings)) + " " + str(type(current_face_encoding)))
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    best_match_index = np.argmin(face_distances)
    #print("best_match_index is " + str(best_match_index) )

    if face_distances[0].all() < 0.65:  #or 0.65
        metadata = known_face_metadata[0]

    return metadata

def all_contraints_satisfies(face_locations, metadata):
    """CHecks all condition to ensure proper proctoring"""
    global time_of_unknown_single_face, time_of_no_faces, time_of_multiple_faces
    if(len(face_locations) > 1):
        if(time_of_multiple_faces is None):
            time_of_multiple_faces = datetime.now()
        print("Warning!! Multiple faces detected")
        time_of_no_faces = None
        time_of_unknown_single_face = None
        if( time_of_multiple_faces is not None and datetime.now() - time_of_multiple_faces > timedelta(seconds=10) ):
            print("Exiting, because multiple faces are in front of camera")
            return False
        else:
            return True
    elif(len(face_locations) <=0 ):
        if time_of_no_faces is None:
            time_of_no_faces = datetime.now()
        print("Don't move away from camera.")
        time_of_multiple_faces = None
        time_of_unknown_single_face = None
        if(time_of_no_faces is not None and datetime.now() - time_of_no_faces > timedelta(seconds = 10)):
            print("Exiting, because no one is in front of camera.")
            return False
        else:
            return True
    else:
        print("correct")
        if metadata is not None:
            return True
        else:
            return False

def main_loop():
    video_capture = cv2.VideoCapture(0)
    global flag
    number_of_faces_since_save = 0

    while True:
        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_labels = []

        for face_location, face_encoding in zip(face_locations, face_encodings):
            metadata = lookup_known_face(face_encoding)
            flag = all_contraints_satisfies(face_locations, metadata)
            print(flag)
            if flag is False:
                break
            if metadata is not None:

                time_at_door = datetime.now() - metadata['last_seen']
                face_label = f"Face Recognized {int(time_at_door.total_seconds())}s"

            else:
                face_label = "New Face!"
                print('Unknown Face')

                top, right, bottom, left = face_location
                face_image = small_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))

                register_new_face(face_encoding, face_image)

            face_labels.append(face_label)
        if flag:
            break
        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (175, 106, 91), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (175, 106, 91), cv2.FILLED)
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, "Faces Recognized", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break

        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            save_known_faces()
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1

    video_capture.release()
    cv2.destroyAllWindows()


while True:
    event, values = window.Read()
    # ret, frame = cap.read()
    if event is None:
        window.Close
        break

    elif event in 'Start':
        image = cv2.imread("E:/Backup/shubham/ME/My Documents/my_pic.jpg", cv2.IMREAD_UNCHANGED)
        if not save_new_face_image(image):
            break
        else:
            #load_known_faces()
            main_loop()

    elif event in ('Exit', None):
        window.Close()
        break

