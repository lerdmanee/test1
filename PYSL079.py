import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp

hands = mp.solutions.hands.Hands(max_num_hands=2)

st.title("Landmark บนฝ่ามือ")

class VideoProcessor:  
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #------------------------------------------------
        img = cv2.flip(img,1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR --> RGB
        results = hands.process(imgRGB)
        
        if results.multi_hand_landmarks: #พบมือในภาพรึเปล่า
            #print(len(results.multi_hand_landmarks))
            for handLms in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img,handLms,
                                                          mp.solutions.hands.HAND_CONNECTIONS)
        #------------------------------------------------
        return av.VideoFrame.from_ndarray(img,format="bgr24")

webrtc_streamer(key="test",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True,"audio": False})
