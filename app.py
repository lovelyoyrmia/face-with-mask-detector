import cv2
import numpy as np
import streamlit as st
import requests
from PIL import Image
from helper_model import detector

@st.cache()
def load_image_pil(image_upload):
    image = Image.open(image_upload)
    return image

def load_image_from_url(image_upload):
    image = None
    try:
        image_url = requests.get(image_upload, stream=True)
        try:
            image = Image.open(image_url.raw)
        except Exception:
            st.error('Cannot upload your image, please try another url')
            image = None
    except Exception:
        st.error('Please input a valid url')
        image = None
    
    return image


def uploader():
    image_upload = None
    image = None

    image_upload_option = st.selectbox('Choose Image Uploader', ['Choose Type', 'Drag and Drop Image', 'Upload From Url'])
    
    if image_upload_option == 'Choose Type':
        image_upload = None
        image = None

    elif image_upload_option == 'Drag and Drop Image':
        image_upload = st.file_uploader(
            'Choose image to predict', ['jpg', 'jpeg', 'png'])
        image = load_image_pil(image_upload)

    else:
        image_url = st.text_input('Put Your URL Link')

        if image_url != '':
            image = load_image_from_url(image_url)

            if image is not None:
                image_upload = image_url

            else:
                image_upload = None

            return image, image_upload

        else:
            image_upload = None

    return image, image_upload


def preprocessing_main():
    selected_features = st.sidebar.radio(
        'Select Features', ['Detect with images', 'Detect with videos'])

    if selected_features == 'Detect with images':
        process_image()

    else:
        process_video()


def set_image(image):
    return st.image(image, use_column_width=True)


def process_image():
    image, image_upload = uploader()

    if image_upload is not None:
        
        if st.sidebar.button('Detect Face Mask'):
            frame = np.array(image.convert('RGB'))
            image_result = detector(frame)
            st.sidebar.subheader('Original Image')
            st.sidebar.image(image, use_column_width=True)
            st.subheader('Result')
            set_image(image_result)
        else:
            st.subheader('Original Image')
            set_image(image)

    else:
        st.info(
            'Upload your image to detect whether it\'s using mask or not')


def process_video():
    run_video = st.checkbox('Run Video')
    frame_window = st.image([])
    cam = cv2.VideoCapture(0)

    try:
        while run_video:
            _, frame = cam.read()
            frame = detector(frame)
            frame_window.image(frame)

            st.error('Cannot load the video')

        else:
            st.info('Run video to detect whether it\'s using mask or not')
    except Exception:
        st.error('Cannot load the video')


# Main App

def about_main():
    st.subheader('About me')
    st.markdown(
        """
            <link
                rel="stylesheet"
                href="https://use.fontawesome.com/releases/v5.15.4/css/all.css"
                integrity="sha384-DyZ88mC6Up2uqS4h/KRgHuoeGwBcD4Ng9SiP4dIRy0EXTlnuz47vAwmeGwVChigm"
                crossorigin="anonymous"
            />
            
            <div class='icon-container'>
                <a href='https://github.com/lovelyoyrmia' target='_blank' rel='noopener noreferrer'><i class='fab fa-github'></i></a>
                <a href='https://www.instagram.com/lovelyoyrmia/' target='_blank' rel='noopener noreferrer'><i class='fab fa-instagram'></i></a>
                <a href='https://www.linkedin.com/in/lovelyoyrmia' target='_blank' rel='noopener noreferrer'><i class='fab fa-linkedin'></i></a>
            </div>
            <div class='link-web'>
                For more info you can click here <a href='https://lovelyoyrmia.github.io' target='_blank' rel='noopener noreferrer'>My Portfolio</a>
            </div>
            <h5>Created with ❤ by Lovelyo Yeremia</h5>
            <style>
            .icon-container a {
                text-decoration: none;
                color: rgb(255, 255, 255);
                font-size: 2rem;
            }
            .link-web {
                font-size: 25px;
                font-weight: 600;
            }
            .link-web a {
                color: rgba(0, 0, 255, 0.5);
                text-decoration: none;
            }
            .icon-container {
                display: flex;
                width: 30%;
                justify-content: space-between;
            }
            .link-web a:hover,
            .icon-cotainer a:hover {
                color: rgba(255, 255, 255, 0.7);
            }
            h5 {
                margin-top: 10px;
            }
            </style>
      """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Mask Detect App", layout="wide")
    hide_menu_style = """
    <style>
        #MainMenu {display: none; }
        footer {visibility: hidden;}
        .css-fk4es0 {display: none;}
        #stStatusWidget {display: none;}
        .css-r698ls {display: none;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.title('Face Mask Detection')
    st.text('Detect Mask using computer vision and deep learning algorithm')

    pages = ['Preprocessing', 'About']
    selected_page = st.sidebar.selectbox('Select Page', pages)

    if selected_page == 'Preprocessing':
        preprocessing_main()
    else:
        about_main()


if __name__ == '__main__':
    main()
