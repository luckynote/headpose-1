#!/usr/bin/env python
# -*- coding:utf-8 -*-

import glob
import os
import numpy as np
import cv2
import re

def getDims(img):
    datapath = os.path.splitext(img)[0] + ".txt"
    file = open(datapath, "r")

    #Descartar líneas
    file.readline()
    file.readline()
    file.readline()

    #Leer datos
    centerx = file.readline()
    centery = file.readline()
    width = file.readline()
    height = file.readline()

    return int(centerx), int(centery), int(width), int(height)

def getSerie(img):
    serie = os.path.basename(img)[8]

    return int(serie)

def getFace(img):

    [centerx, centery, width, height] = getDims(img)
    xcorner = (centerx - int(width/2))
    ycorner = (centery - int(height/2))

    pic = cv2.imread(img, cv2.IMREAD_GRAYSCALE)[ycorner:(ycorner + height), xcorner:(xcorner + width)]

    return pic

def getValues(img):

    raw = os.path.basename(img)[11:]
    raw = os.path.splitext(raw)[0]

    tilt, pan = re.split('[-+]', raw[1:])

    tilt = raw[0] + tilt
    pan = raw[len(raw) - len(pan) - 1] + pan

    return int(tilt), int(pan)

def getData(width, height):
    face_size = width * height

    serie1 = np.zeros((1395, face_size))
    serie2 = np.zeros((1395, face_size))
    values1 = np.zeros((1395, 2))
    values2 = np.zeros((1395, 2))

    i1 = 0
    i2 = 0
    for i in range(1, 16):
        num = '{0:02}'.format(i)
        path = 'HeadPoseImageDatabase/Person' + num
        imagenes = glob.glob(path + "/*.jpg")

        for img in imagenes:
            if getSerie(img) == 1:
                pic = getFace(img)
                pic = cv2.resize(pic, (width, height))
                flat = pic.flatten() / 255.0
                serie1[i1] = flat
                values1[i1] = getValues(img)
                i1 += 1

            else:
                pic = getFace(img)
                pic = cv2.resize(pic, (width, height))
                flat = pic.flatten() / 255.0
                serie2[i2] = flat
                values2[i2] = getValues(img)
                i2 += 1

    return serie1, values1, serie2, values2