import cv2
import numpy as np
from keras.models import load_model


def load_mymodel():
    model = load_model('model/model_facemask.h5')
    model_hrc = 'model/haarcascade_frontalface_default.xml'

    return model, model_hrc


def detector(frame):
    model, model_hrc = load_mymodel()
    frame = cv2.cvtColor(frame, 1)
    faceCascade = cv2.CascadeClassifier(model_hrc)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 6)
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        faces2 = faceCascade.detectMultiScale(roi_gray, 1.1, 6)
        if len(faces2) == 0:
            print('Face not detected')
        else:
            for (ex, ey, ew, eh) in faces2:
                face_roi = roi_color[ey: ey+eh, ex:ex+ew]

        predictions = predict_image(frame, model, face_roi)

        font = cv2.FONT_HERSHEY_COMPLEX
        if w < 100:
            font_scale = 0.45
            width_rect = 0
            height_rect = 30
            font_weight = 1
        else:
            font_scale = 1
            width_rect = 30
            height_rect = 40
            font_weight = cv2.LINE_4

        if (predictions > 0):
            status = 'No Mask!'

            cv2.rectangle(frame, (x, y), (x+w-width_rect, y-height_rect),
                          (255, 255, 255), -1)
            cv2.rectangle(frame, (x, y), (x+w-width_rect, y-height_rect),
                          (255, 0, 0), 2)
            cv2.putText(frame, status, (x+8, y-8), font,
                        font_scale, (255, 0, 0), 2, font_weight)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            status = 'Face Mask'

            cv2.rectangle(frame, (x, y), (x+w-width_rect, y-height_rect),
                          (255, 255, 255), -1)
            cv2.rectangle(frame, (x, y), (x+w-width_rect, y-height_rect),
                          (0, 255, 0), 2)
            cv2.putText(frame, status, (x+8, y-8), font,
                        font_scale, (0, 255, 0), 2, font_weight)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame


def predict_image(frame, model, face_roi):
    final_image = cv2.resize(face_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image/255

    predictions = model.predict(final_image)
    print(predictions)
    return predictions

# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()

#     detector(frame)

#     cv2.imshow('', frame)

#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break
