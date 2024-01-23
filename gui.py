import tkinter as tk
from tkinter import filedialog
from tkinter import *
from sklearn import metrics
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from functions_s import extract


def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss = 'categorical_entropy', metrics=['accuracy'])
    return model


def SpeechExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model


top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')
label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
facec = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model_f = FacialExpressionModel("model_a1_f.json","model_weights1.h5")
model_s = SpeechExpressionModel("model_a1_s.json","model_weights4.h5")
EMOTIONS_LIST_f = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
EMOTIONS_LIST_s = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']


def clear_interface():
    for widget in top.winfo_children():
        if widget not in [emotion_f, emotion_s]:
            widget.destroy()


def Detect_f(file_path):
    global label1
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    try:
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,(48,48))
            pred = EMOTIONS_LIST_f[np.argmax(model_f.predict(roi[np.newaxis,:,:,np.newaxis]))]
        dis = "Predicted Emotion : " + pred
        label1.configure(foreground="#011638", text=dis)
    except:
        label1.configure(foreground="#011638", text="Unable to detect")


def Detect_s(file_path):
    try:
        import numpy as np
        features = extract(file_path, mfcc=True, chroma=True, mel=True)
        features_reshaped = np.expand_dims(features, axis=0)
        emotion_index = np.argmax(model_s.predict(features_reshaped))
        pred = EMOTIONS_LIST_s[emotion_index]
        dis = "Predicted Emotion : " + pred
        label1.configure(foreground="#011638", text=dis)
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        label1.configure(foreground="#011638", text="Unable to detect")


def show_Detect_button_f(file_path):
    detect_b = Button(top, text="Detect Emotion", command=lambda:Detect_f(file_path),padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial',10,'bold'))
    detect_b.place(relx=0.79, rely=0.46)


def show_Detect_button_s(file_path):
    detect_b = Button(top, text="Detect Emotion", command=lambda:Detect_s(file_path),padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial',10,'bold'))
    detect_b.place(relx=0.79, rely=0.46)


def upload_image():
    global sign_image, label1, im
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button_f(file_path)
        upload_image.photo = im
    except Exception as e:
        print(f"Error is {e} ")
        pass


def upload_audio():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Audio files","*.wav")])
        file_name = os.path.basename(file_path)
        label1.configure(foreground="#011638", text=f"Uploaded Audio: {file_name}")
        show_Detect_button_s(file_path)
    except:
        pass

def speech():
    global label1, top
    clear_interface()
    top.geometry('800x600')
    top.title('Emotion Detector')
    top.configure(background='#CDCDCD')
    label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
    upload = Button(top, text="Upload Audio", command=upload_audio, padx=10, pady=5)
    upload.configure(background="#364156", foreground='white', font=('arial',20,'bold'))
    upload.pack(side='bottom', pady=50)
    label1.pack(side='bottom', expand='True')
    heading = Label(top, text='Emotion Detector', pady=20, font=('arial',25,'bold'))
    heading.configure(background="#CDCDCD", foreground="#364156")
    heading.pack()
    top.mainloop()


def facial():
    global sign_image, label1
    clear_interface()
    top.geometry('800x600')
    top.title('Emotion Detector')
    top.configure(background='#CDCDCD')
    label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
    sign_image = Label(top)
    upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
    upload.configure(background="#364156", foreground='white', font=('arial',20,'bold'))
    upload.pack(side='bottom', pady=50)
    sign_image.pack(side='bottom', expand='True')
    label1.pack(side='bottom', expand='True')
    heading = Label(top, text='Emotion Detector', pady=20, font=('arial',25,'bold'))
    heading.configure(background="#CDCDCD", foreground="#364156")
    heading.pack()


emotion_f = Button(top, text="Facial Detection", command=facial, padx=10, pady=5)
emotion_s = Button(top, text="Speech Detection", command=speech, padx=10, pady=5)

emotion_f.pack(side='bottom', pady=10)
emotion_s.pack(side='bottom',pady=10)
top.mainloop()
