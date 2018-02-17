from keras.models import load_model
from preprocess import getFace, getValues, getDims, getData
from others import normalize_set
import cv2
import numpy as np
from math import sin, cos, radians

def main():

    # Definir variables
    width = 46
    height = 46
    avg = 0.48315518468433094
    std = 0.17873064673404526
    face_size = width * height
    maxlen = 50

    # Cargar modelo

    model = load_model("Modelos/modelo7.h5")

    # Cargar imagen

    path = input("Introduzca la ruta de la imagen: ")
    ori = cv2.imread(path, cv2.IMREAD_COLOR)
    pic = getFace(path)

    # Procesar valores para ser usados

    pic = cv2.resize(pic, (width, height))
    flat = pic.flatten() / 255.0
    t_img = np.zeros((1, face_size))
    t_img[0] = flat
    t_img = np.reshape(t_img, (t_img.shape[0], height, width, 1))

    normalize_set(t_img, avg, std) # Es necesario?

    # Obtener valores reales y estimados

    [t_tilt, t_pan] = getValues(path)
    pred = model.predict(t_img)

    pred[:,0] = pred[:,0]*90
    pred[:,1] = pred[:,1]*90

    [centerx, centery, width, height] = getDims(path)

    t_offsetx = -1 * int(sin(radians(t_pan)) * maxlen)
    t_offsety = -1 * int(sin(radians(t_tilt)) * maxlen)

    e_offsetx = -1 * int(sin(radians(pred[0,1])) * maxlen)
    e_offsety = -1 * int(sin(radians(pred[0,0])) * maxlen)

    print("Real: %f %f" % (t_tilt, t_pan))
    print("Estimado: %f %f" % (pred[0,0], pred[0,1]))

    # Obtener puntos

    center = (centerx, centery)
    t_end = (centerx + t_offsetx, centery + t_offsety)
    e_end = (centerx + e_offsetx, centery + e_offsety)

    # Dibujar flechas

    cv2.arrowedLine(ori, center, t_end, (0, 0, 255), 2)
    cv2.arrowedLine(ori, center, e_end, (0, 255, 0), 2)

    # Mostrar imagen

    cv2.imshow("Detectado", ori)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()