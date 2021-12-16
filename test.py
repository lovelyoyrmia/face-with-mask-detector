import cv2
import numpy as np
from keras.models import load_model


model = load_model('model/model_facemask.h5')

# Declare Variable
path = 'model/haarcascade_frontalface_default.xml'
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# Set rectangle background
rectangle_bgr = (255, 255, 255)
# Make black image
img = np.zeros((500, 500))
# Set text
text = ''
# Get Width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, font_scale, 1)[0]

# Set Start position
text_offset_x = 10
text_offset_y = img.shape[0] - 15

# Make the coordinates of the box
box_coord = ((text_offset_x, text_offset_y), (text_offset_x +
             text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coord[0], box_coord[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y),
            font, font_scale, (0, 0, 0), 1)

# cap = cv2.VideoCapture(0)

while True:
    frame = cv2.imread('vio1.jpg')
    faceCascade = cv2.CascadeClassifier(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 5)
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        faces2 = faceCascade.detectMultiScale(roi_gray, 1.1, 5)
        if len(faces2) == 0:
            print('Face not detected')
        else:
            for (ex, ey, ew, eh) in faces2:
                face_roi = roi_color[ey: ey+eh, ex:ex+ew]
    final_image = cv2.resize(face_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image/255
    font = cv2.FONT_HERSHEY_SIMPLEX

    predictions = model.predict(final_image)
    font_scale = 1.5

    if (predictions > 0):
        status = 'No Mask'

        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 +
                    int(h1/2)), font, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, status, (100, 150), font,
                    2, (0, 0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
    else:
        status = 'Face Mask'

        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 +
                    int(h1/2)), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, status, (100, 150), font,
                    2, (0, 255, 0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))

    cv2.imshow('', frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
