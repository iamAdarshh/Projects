import cv2
import numpy as np

detection = False
framecounter = 0

# Recorded Video
cap = cv2.VideoCapture('recordedVideo.mp4')

# Target Image and video.
targetImage = cv2.imread('TargetImage.jpg')
targetVideo = cv2.VideoCapture('targetVideo1.mp4')

success, videoImg = targetVideo.read()

# Getting Target Image size to make video size same.
height, width, channels = targetImage.shape
videoImg = cv2.resize(videoImg, (width, height))

# Detector
orb = cv2.ORB_create(nfeatures=1000)
# Keypoints and Descriptors.
kp1, des1 = orb.detectAndCompute(targetImage, None)
# targetImage = cv2.drawKeypoints(targetImage, kp1, None)

while True:
    success, imgrecorded = cap.read()
    imgrecorded = cv2.resize(imgrecorded, (width, height))

    imgAug = imgrecorded.copy()

    # Finding keypoints and descriptors for webcam Image.
    kp2, des2 = orb.detectAndCompute(imgrecorded, None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

    if detection == False:
        targetVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
        framecounter = 0
    else:
        if framecounter == targetVideo.get(cv2.CAP_PROP_FRAME_COUNT):
            targetVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
            framecounter = 0
        success, videoImg = targetVideo.read()
        videoImg = cv2.resize(videoImg, (width, height))
        

    # Matching descriptors with BruteForce Method.
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    # Printing Number of Good Matches.
    print(len(good))

    imgFeatures = cv2.drawMatches(targetImage, kp1, imgrecorded, kp2, good, None, flags=2)

    if len(good)>20:
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)

        img2 = cv2.polylines(imgrecorded, [np.int32(dst)], True, (255, 0, 255), 3)

        imgWarp = cv2.warpPerspective(videoImg, matrix, (imgrecorded.shape[1], imgrecorded.shape[0]))

        maskNew = np.zeros((imgrecorded.shape[0], imgrecorded.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
        maskInv = cv2.bitwise_not(maskNew)

        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

    cv2.imshow('Mask New', imgAug)
    #cv2.imshow('Image Wrap', imgWarp)
    cv2.imshow('imgFeatures', imgFeatures)
    #cv2.imshow('targetImage', targetImage)
    cv2.imshow('targetVideo', videoImg)
    cv2.imshow('Recorded Image', imgrecorded)
    #cv2.imshow('Img 2', img2)
    cv2.waitKey(1)
    framecounter+=1