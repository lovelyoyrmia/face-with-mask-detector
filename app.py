import cv2
import streamlit as st
from PIL import Image
from helper_model import detector


def preprocessing_main():
    selected_features = st.sidebar.radio(
        'Select Features', ['Detect with images', 'Detect with videos'])

    if selected_features == 'Detect with images':
        process_image()

    else:
        process_video()


def process_image():
    image_upload = st.file_uploader(
        'Choose image to predict', ['jpg', 'jpeg', 'png'])

    if image_upload is not None:
        image = load_image_pil(image_upload)

        st.subheader('Original Image')
        st.image(image)

        if st.sidebar.button('Detect Face Mask'):
            pass

    else:
        st.warning(
            'Upload your image to detect whether it\'s using mask or not')


def process_video():
    run_video = st.checkbox('Run Video')
    frame_window = st.image([])
    cam = cv2.VideoCapture(0)

    while run_video:
        _, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)
        detector(frame)

    else:
        st.warning('Stopped Video')
# Main App


@st.cache()
def load_image_pil(image_upload):
    image = Image.open(image_upload)
    return image


def main():
    st.title('Face Masked Detection')

    pages = ['Preprocessing', 'About']
    selected_page = st.sidebar.selectbox('Select Page', pages)

    if selected_page == 'Preprocessing':
        preprocessing_main()


if __name__ == '__main__':
    main()
