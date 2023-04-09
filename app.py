import smtplib
import streamlit as st
import numpy as np
import keras
import cv2
from PIL import Image
import os
from add_gmail import add_gmail


## STREAMLIT APP

st.title('Gastrointestional Cancer Detection System')
reciever = st.text_input(label='Enter your email')
uploaded_file = st.file_uploader("Upload Your Endoscopic Report",accept_multiple_files=False)

st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url('https://wallpapers.com/images/hd/mbbs-blue-caduceus-symbol-lc782hzpjuqtbax1.jpg');
             background-attachment: fixed;
             background-size: contain 
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

if uploaded_file != None:
        with open(os.path.join("C:/users/Public/",uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success("Saved file")
        path = "C:/users/Public/"+str(uploaded_file.name)
        image = cv2.imread(path)
        image_show = Image.open(path)
        image_show = image_show.resize((500,250))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_image,(200,200), interpolation = cv2.INTER_AREA)
        y = np.expand_dims(resized, axis=-1)
        y = np.expand_dims(y,axis=0)
        automl = keras.models.load_model('D:/model_CNN.h5')
        predictions = automl.predict(y).flatten()
        if int(predictions[0]) == 1:
            new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Cyst</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            st.image(image_show)
            gmail = add_gmail(reciever,type_cancer='cyst')
            gmail.gmail_send()
        elif int(predictions[1]) == 1:
            new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Normal</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            st.image(image_show)
            gmail = add_gmail(reciever, type_cancer='Normal')
            gmail.gmail_send()
        elif int(predictions[2]) == 1:
            new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Stone</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            st.image(image_show)
            gmail = add_gmail(reciever, type_cancer='Stone')
            gmail.gmail_send()
        elif int(predictions[3]) == 1:
            new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Tumor</p>'
            st.markdown(new_title, unsafe_allow_html=True)
            st.image(image_show)
            gmail = add_gmail(reciever, type_cancer='Tumor')
            gmail.gmail_send()
else :
        pass