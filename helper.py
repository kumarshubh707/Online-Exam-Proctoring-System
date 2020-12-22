import PySimpleGUI as sg
import os.path
import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle


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

known_face_encoding = []
known_face_metadata = []

def click_image():
    image  = None
    vid = cv2.VideoCapture(0)
    print("Press 's' to click photo..")
    while (True):
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            image = frame
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
    return image

def lookup_known_face(face_encoding):
    """
    See if this is a face we already have in our face list
    """

    metadata = None

    if len(known_face_encoding) == 0:
        return metadata
    face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)

    best_match_index = np.argmin(face_distances)

    if face_distances[best_match_index] < 0.65:

        metadata = known_face_metadata[best_match_index]

    return metadata


def main_loop(image):
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    if (len(face_locations) != 1 and len(face_encodings) != 1):
        print("Image not valid, Kindly select new image")
        exit(0)

    top, right, bottom, left = face_locations[0]
    face_image = small_frame[top:bottom, left:right]
    face_image = cv2.resize(face_image, (150, 150))
    known_face_encoding.append(face_encodings[0])
    known_face_metadata.append({
        "first_seen": datetime.now(),
        "last_seen": datetime.now(),
        "face_image": face_image,
    })

    time_of_multiple_faces = None
    time_of_no_faces = None
    time_of_unknown_single_face = None
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # face_labels = []
        number_of_faces = len(face_encodings)
        if number_of_faces == 1:
            metadata = False
            face_distances = face_recognition.face_distance(known_face_encoding, face_encodings[0])
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.65:  # or 0.65
                metadata = True

            if metadata:
                print("Correct face")
                time_of_multiple_faces=None
                time_of_unknown_single_face=None
                time_of_no_faces = None
            else:
                print('Unknown Face detected')
                time_of_multiple_faces=None
                time_of_no_faces = None
                if time_of_unknown_single_face is not None and datetime.now() - time_of_unknown_single_face > timedelta(seconds=15):
                    print("Exiting, because unknown user detected!!")
                    exit(0)
                elif time_of_unknown_single_face is None:
                    time_of_unknown_single_face = datetime.now()

        elif number_of_faces > 1:
            time_of_no_faces = None
            time_of_unknown_single_face = None
            print("Warning!! Multiple faces detected")
            if time_of_multiple_faces is not None and datetime.now() - time_of_multiple_faces > timedelta(seconds=10):
                print("Exiting, because multiple faces are in front of camera")
                exit(0)
            elif time_of_multiple_faces is None:
                time_of_multiple_faces=datetime.now()
        else:
            time_of_unknown_single_face = None
            time_of_multiple_faces = None
            print("Don't move away from camera.")
            if time_of_no_faces is not None and datetime.now() - time_of_no_faces > timedelta(seconds=10):
                print("Exiting, because no one is in front of camera.")
                exit(0)
            elif time_of_no_faces is None:
                time_of_no_faces = datetime.now()

        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (175, 106, 91), 2)
            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (175, 106, 91), cv2.FILLED)
            #cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    while True:
        event, values = window.Read()
        # ret, frame = cap.read()
        if event is None:
            window.Close
            break

        elif event in 'Start':
            image = click_image()
            #cv2.imread("E:/Backup/shubham/ME/My Documents/my_pic.jpg", cv2.IMREAD_UNCHANGED)
            if(image is not None):
                main_loop(image)
            else:
                print("Failed to load")

        elif event in ('Exit', None):
            window.Close()
            break

