import numpy as np
import cv2
from keras.models import load_model
from others import normalize_set
from math import sin, radians

# Declarar valores

width = 40
height = 40
avg = 0.48315518468433094
std = 0.17873064673404526
face_size = width * height
maxlen = 200
model = load_model("Modelos/modelo1.h5")

# Detectar caras

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = input("Introduzca la ruta de la imagen: ")
img = cv2.imread(path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for(x, y, w, h) in faces:
    pic = gray[y:y+h, x:x+w]

    pic = cv2.resize(pic, (width, height))

    flat = pic.flatten() / 255.0
    t_img = np.zeros((1, face_size))
    t_img[0] = flat
    t_img = np.reshape(t_img, (t_img.shape[0], height, width, 1))

    normalize_set(t_img, avg, std)

    pred = model.predict(t_img)

    pred[:, 0] = pred[:, 0] * 90
    pred[:, 1] = pred[:, 1] * 90

    e_offsetx = -1 * int(sin(radians(pred[0, 1])) * maxlen)
    e_offsety = -1 * int(sin(radians(pred[0, 0])) * maxlen)

    centery = int((2*y+h)/2)
    centerx = int((2*x+w)/2)

    center = (centerx, centery)
    e_end = (centerx + e_offsetx, centery + e_offsety)

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    cv2.arrowedLine(img, center, e_end, (0, 255, 0), 2)

cv2.imshow("Detectado", img)
cv2.waitKey(0)