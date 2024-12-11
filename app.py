import streamlit as st
from utils import *
import cv2
import numpy as np

def main(weights_path: str):
    print("Hello World")
    st.title("Fire Smoke Detection - YOLOv11 - Ultralytics")
    st.header("Upload your image below !")

    # Upload image file
    uploaded_file = st.file_uploader("Choose your file:")

    if uploaded_file is not None:
        # Load image
        image_bytes = uploaded_file.getvalue()
        orig_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        st.subheader("Original Image")
        st.image(orig_image, caption="Original Image", use_column_width=True)

        # Load model 
        model = load_model(weights_path)

        # Predict and detect image
        prediction = predict_and_detect(pretrained_model=model,
                                        file=orig_image)
        
        # Show prediction result
        st.subheader("Detected Object")
        st.image(prediction[0], caption="Detection Object", use_column_width=True)


if __name__ == '__main__':
    weights_path = './weights/best.pt'
    main(weights_path=weights_path)