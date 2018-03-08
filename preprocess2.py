from preprocess import getDims, getSerie, getValues
import cv2
import glob
import numpy as np
import random

def getFace2(img, alter = 0):

    [centerx, centery, width, height] = getDims(img)

    xcorner = (centerx - int(width / 2))
    ycorner = (centery - int(height / 2))

    if(alter == 0):
        pic = cv2.imread(img, cv2.IMREAD_GRAYSCALE)[ycorner:(ycorner + height), xcorner:(xcorner + width)]
    else:

        x_offset = int((width * random.uniform(5, 10)) / 100)
        y_offset = int((height * random.uniform(5, 10)) / 100)

        if random.randint(0, 1) == 1:
            x_offset = x_offset * -1

        if random.randint(0, 1) == 1:
            y_offset = y_offset * -1

        pic = cv2.imread(img, cv2.IMREAD_GRAYSCALE)[ycorner:(ycorner + height + y_offset), xcorner:(xcorner + width + x_offset)]

    return pic

def getData2(width, height, numvars):

    face_size = width * height
    vect_size = 1395 * numvars * 2

    serie1 = np.zeros((vect_size, face_size))
    serie2 = np.zeros((vect_size, face_size))
    values1 = np.zeros((vect_size, 2))
    values2 = np.zeros((vect_size, 2))

    i1 = 0
    i2 = 0
    facecount = 0

    # Para todas las imagenes del set

    for i in range(1, 16):
        num = '{0:02}'.format(i)
        path = 'HeadPoseImageDatabase/Person' + num
        imagenes = glob.glob(path + "/*.jpg")

        for img in imagenes:

            print("Cargando imagen %d..." % facecount)

            # Obtener variaciones de la imagen.

            for i in range(0, numvars):

                pic = getFace2(img, i)
                pic_eq = cv2.equalizeHist(pic)
                pic = cv2.resize(pic, (width, height))
                pic_eq = cv2.resize(pic_eq, (width, height))
                flat1 = pic.flatten() / 255.0
                flat2 = pic_eq.flatten() / 255.0
                values = getValues(img)

                if getSerie(img) == 1:
                    serie1[i1] = flat1
                    serie1[i1 + 1] = flat2
                    values1[i1] = values
                    values1[i1 + 1] = values
                    i1 += 2
                else:
                    serie2[i2] = flat1
                    serie2[i2 + 1] = flat2
                    values2[i2] = values
                    values2[i2 + 1] = values
                    i2 += 2

            facecount += 1

    return serie1, values1, serie2, values2

