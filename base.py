import face_recognition
from tkinter import Label
from cv2.cv2 import VideoCapture
from PIL import Image, ImageTk
from idlelib import window
from tkinter.ttk import *
from tkinter import *
import numpy as np
import pickle
import PIL
import cv2
import os

FACE_DECT = Tk()
FACE_DECT.title("FACE DETECTION FOR HOME SECURITY")

GEOMETRY = str(FACE_DECT.winfo_screenwidth() - 500) + "x" + str(FACE_DECT.winfo_screenheight() - 260)
FACE_DECT.geometry(GEOMETRY)

mframe = Frame(FACE_DECT, highlightbackground="green", highlightcolor="green", highlightthickness=5, width=(FACE_DECT.winfo_screenwidth() - 500), height=(FACE_DECT.winfo_screenheight() - 260),
               bd=0)
mframe.place(x=0, y=0)
frame1 = Frame(FACE_DECT, highlightbackground="green", highlightcolor="green", highlightthickness=5, width=(FACE_DECT.winfo_screenwidth()-881), height=(FACE_DECT.winfo_screenheight()-369),
               bd=0)
frame1.place(x=0, y=0)
FACE_DECT.bind('<Escape>', lambda e: FACE_DECT.quit())
lmain: Label = Label(FACE_DECT)
lmain.place(x=5, y=5)
take_photo_xbtn = 10
take_photo_ybtn = (FACE_DECT.winfo_screenheight()-364)
take_photo_xbtn = 10
take_photo_ybtn = (FACE_DECT.winfo_screenheight()-364)
frame2 = Frame(FACE_DECT, highlightbackground="green", highlightcolor="green", highlightthickness=5, width=(FACE_DECT.winfo_screenwidth()-1253), height=(FACE_DECT.winfo_screenheight()-770),
               bd=0)
frame2.place(x=take_photo_xbtn-2, y=take_photo_ybtn-2)
style = Style()

btn = Button(FACE_DECT, text='EXIT', font = ('calibri', 30, 'bold', 'underline'),foreground = 'red', command = FACE_DECT.destroy)
btn.place(x = take_photo_xbtn+3, y = take_photo_ybtn+3)

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 1
FONT_THICKNESS = 1
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, (FACE_DECT.winfo_screenwidth()-(FACE_DECT.winfo_screenwidth()/2)))
video.set(cv2.CAP_PROP_FRAME_HEIGHT, (FACE_DECT.winfo_screenheight()-(FACE_DECT.winfo_screenheight()/2)))

# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


#print('Loading known faces...')
known_faces = []
known_names = []

# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)


#print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
#while True:
def show_frame():
    # Load image
    #print(f'Filename {filename}', end='')
    #image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
    ret, image = video.read()

    # This time we first grab face locations - we'll need them to draw boxes
    locations = face_recognition.face_locations(image, model=MODEL)

    # Now since we know loctions, we can pass them to face_encodings as second argument
    # Without that it will search for faces once again slowing down whole process
    encodings = face_recognition.face_encodings(image, locations)

    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    #print(f', found {len(encodings)} face(s)')
    #cv2.rectangle(image, top_left, bottom_right, (0,255,0), FRAME_THICKNESS)
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Since order is being preserved, we check if any face was found then grab index
        # then label (name) of first matching known face withing a tolerance
        match = None
        if True in results:  # If at least one is true, get a name of first of found labels
            match = known_names[results.index(True)]
            #print(f' - {match} from {results}')

            # Each location contains positions in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = [0,255,0]

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame below for a name
            # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Wite a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), FONT_THICKNESS)

    # Show image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #cv2.imshow("filename", image)
    img = PIL.Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break
show_frame()
FACE_DECT.mainloop()
    #cv2.waitKey(0)
    #cv2.destroyWindow(filename)
