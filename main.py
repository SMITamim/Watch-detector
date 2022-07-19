import cv2
import matplotlib as mt
import numpy as np

img01 = cv2.imread('watch01.jpg',0) #importing image as grayscale
img02 = cv2.imread('watch02.jpg',0)

orb = cv2.ORB_create() #orb algo is the fastest one and free to use amoung others

key_point01, descriptor01 = orb.detectAndCompute(img01,None) #keypoints for image 01
key_point02, descriptor02 = orb.detectAndCompute(img02,None) #keypoints for image 02

bf = cv2.BFMatcher()                                         #brute force matcher will match a key point with all the other 500 keypoints
matches = bf.knnMatch(descriptor01,descriptor02,k=2)



#imgKey_point01 = cv2.drawKeypoints(img01,key_point01,None) #printing keypoints for img 01 & 2
#imgKey_point02 = cv2.drawKeypoints(img02,key_point02,None)

#cv2.imshow('key point 01', imgKey_point01)
#cv2.imshow('key point 02', imgKey_point02)
cv2.imshow('img1',img01)
cv2.imshow('img2',img02)
cv2.waitKey(0)
