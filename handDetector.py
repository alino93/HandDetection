import cv2 as cv
from cvzone.HandTrackingModule import HandDetector

cap = cv.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # With Draw

    if hands:
        # first hand
        hand1 = hands[0]
        lm1 = hand1["lmList"]  # List of 21 Landmarks points
        bbox1 = hand1["bbox"]  # Bounding Box info x,y,w,h
        centerPt1 = hand1["center"]  # center of the hand cx,cy
        hType1 = hand1["type"]  # Hand Type Left or Right

        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            # second hand
            hand2 = hands[1]
            lm2 = hand2["lmList"]  # List of 21 Landmarks points
            bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
            centerP2 = hand2["center"]  # center of the hand cx,cy
            hType2 = hand2["type"]  # Hand Type Left or Right

            fingers2 = detector.fingersUp(hand2)
            # print(fingers1, fingers2)
            # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img) # with draw
            length, info, img = detector.findDistance(centerPt1, centerPt2, img)  # with draw

    cv.imshow("Image", img)
    cv.waitKey(1)