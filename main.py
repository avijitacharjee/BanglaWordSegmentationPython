from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
# from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image
from numpy import *


def main():
    print('Hello AI')
    # image = cv2.imread('WordInputFile/WordPage000000.bmp')
    # cv2.imshow("image",image)
    # cv2.waitKey()

    size = (1000, 400)
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    threshold = .3
    data = []
    labels = []

    pos_im_path = r"./Data/pos"
    neg_im_path = r"./Data/neg"

    pos_im_listing = os.listdir(pos_im_path)
    neg_im_listing = os.listdir(neg_im_path)
    num_pos_samples = len(pos_im_listing)
    num_neg_samples = len(neg_im_listing)
    print(num_pos_samples, num_neg_samples)

    le = LabelEncoder()
    print('Reading positive data....')
    for file in pos_im_listing:
        img = cv2.imread(pos_im_path + '/' + file) 
        img = cv2.resize(img, size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fd = hog(img, orientations, pixels_per_cell, cells_per_block,
                 block_norm='L2', feature_vector=True)
        data.append(fd)
        # cv2.imshow('abcd',fd)
        # cv2.waitKey()
        labels.append(1)
    print('Reading negative data....')
    for file in neg_im_listing:
        img = cv2.imread(neg_im_path + '/' + file)
        img = cv2.resize(img, size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fd = hog(img, orientations, pixels_per_cell, cells_per_block,
                 block_norm='L2', feature_vector=True)
        data.append(fd)
        labels.append(0)
    print(len(data))

    print(" Constructing training/testing split...")
    trainData, testData, trainLabels, testLabels = train_test_split(
        data, labels, test_size=0.20, random_state=42)

    #print(trainData[0])
    #print("Label = ", trainLabels[50])

    trainData = np.array(trainData)
    testData = np.array(testData)
    trainLabels = np.array(trainLabels)
    testLabels = np.array(testLabels)

    print(" Training Linear SVM classifier..")
    model = LinearSVC()
    model.fit(trainData, trainLabels)
    pred = model.predict(testData)

    print(classification_report(testLabels, pred, labels=model.classes_,
                                target_names=None, sample_weight=None, digits=3))
    crop(model)


def crop(model):
    size = (1000, 1700)
    size1 = (1000, 400)
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    threshold = .3
    img = cv2.imread('./Data/ddd/imageread/100_13.jpg')
    cv2.imshow("abc", img)
    cv2.waitKey()
    for i in range(0, 1700-100, 30):
        x1, x2, y1, y2 = 0, 1000, i, i + 10
        img = cv2.resize(img, size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop_img = img[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, size1)
        # cv2.imshow("abc", crop_img)
        # cv2.waitKey()
        fd = hog(crop_img, orientations, pixels_per_cell,cells_per_block, block_norm='L2', feature_vector=True)
        pred = model.predict([fd])
        if (pred == 1):
            cv2.imshow('imgj', crop_img)
            cv2.waitKey()
        else:
            print(i)
    print("Crop Image Ended")

if (__name__ == '__main__'):
    main()
#    crop()
