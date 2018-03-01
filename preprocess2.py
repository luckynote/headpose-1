from preprocess import getDims, getSerie, getValues
import cv2
import glob
import numpy as np

def getFace2(img, off_perc, dir):

    [centerx, centery, width, height] = getDims(img)

    if dir == 1 or dir == 2:
        offset = int((off_perc * width)/100)
    else:
        offset = int((off_perc * height)/100)

    xcorner = (centerx - int(width / 2))
    ycorner = (centery - int(height / 2))

    if(dir == 1):
        pic = cv2.imread(img, cv2.IMREAD_GRAYSCALE)[ycorner:(ycorner + height), xcorner:(xcorner + width - offset)]
    elif(dir == 2):
        pic = cv2.imread(img, cv2.IMREAD_GRAYSCALE)[ycorner:(ycorner + height), xcorner:(xcorner + width + offset)]
    elif(dir == 3):
        pic = cv2.imread(img, cv2.IMREAD_GRAYSCALE)[ycorner:(ycorner + height + offset), xcorner:(xcorner + width)]
    elif(dir == 4):
        pic = cv2.imread(img, cv2.IMREAD_GRAYSCALE)[ycorner:(ycorner + height - offset), xcorner:(xcorner + width)]
    else:
        pic = cv2.imread(img, cv2.IMREAD_GRAYSCALE)[ycorner:(ycorner + height), xcorner:(xcorner + width)]

    return pic

def getData2(width, height):
    face_size = width * height

    serie1 = np.zeros((13950, face_size))
    serie2 = np.zeros((13950, face_size))
    values1 = np.zeros((13950, 2))
    values2 = np.zeros((13950, 2))

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

            # Obtener imagen original

            pic = getFace2(img, 0, 0)
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

            # Obtener variaciones de la imagen.

            for i in range(1, 5):

                pic = getFace2(img, 5, i)
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

