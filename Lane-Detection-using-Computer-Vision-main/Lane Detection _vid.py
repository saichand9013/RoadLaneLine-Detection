#Jaimin Kachhadiya
#Aug. 2021
#Detecting lanes in a video

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import cv2

cap = cv2.VideoCapture('C:\VSc\Test\test.mp4')


while(cap.isOpened()):

    
    ret, frame = cap.read()
    cv2.imshow("Original Scene", frame)

    # snip section of video frame of interest & show on screen
    snip = frame[0:1080,0:1920]
    #cv2.imshow("Snip",snip)

    # create polygon (trapezoid) to mask the selected region of interest
    mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
    pts = np.array([[180, 697], [580, 370],[707, 382], [1177, 700]], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    #cv2.imshow("Mask", mask)
    print(snip.shape)
    #apply mask and show masked image on screen
    masked = cv2.bitwise_and(snip, snip, mask=mask)
    #plt.imshow(masked)
    #plt.show()

    # converting to grayscale then black/white to binary image
    frame = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    thresh = 190
    frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Black/White", frame)

    # blurring image to help with edge detection
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    # cv2.imshow("Blurred", blurred)

    # identifying edges 
    edged = cv2.Canny(blurred, 190, 220)
    cv2.imshow("Edged", edged)

    # performing Hough Transform to identify lane lines
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 60, lines=np.array([]))

    # defining arrays for the left and right lanes
    rho_left = []
    theta_left = []
    rho_right = []
    theta_right = []

    # ensuring Hough Lines found at least one line
    if lines is not None:

        # loop through all of the lines found by cv2.HoughLines
        for i in range(0, len(lines)):

            # evaluating each row of the Hough Lines output 'lines'
            for rho, theta in lines[i]:

                # collect points of left lanes
                if theta < np.pi/2 and theta > np.pi/4:
                    rho_left.append(rho)
                    theta_left.append(theta)


                # collect points of right lanes
                if theta > np.pi/2 and theta < 3*np.pi/4:
                    rho_right.append(rho)
                    theta_right.append(theta)


    # identifying the median lane dimensions
    left_rho = np.average(rho_left)
    left_theta = np.average(theta_left)
    right_rho = np.average(rho_right)
    right_theta = np.average(theta_right)

    # plotting median lane 
    if left_theta > np.pi/4:
        a = np.cos(left_theta); b = np.sin(left_theta)
        x0 = a * left_rho; y0 = b * left_rho
        offset1 = 200; offset2 = 300
        x1 = int(x0 - offset1 * (-b)); y1 = int(y0 - offset1 * (a))
        x2 = int(x0 + offset2 * (-b)); y2 = int(y0 + offset2 * (a))

        cv2.line(snip, (x1, y1), (x2, y2), (0, 255, 0), 6)

    if right_theta > np.pi/4:
        a = np.cos(right_theta); b = np.sin(right_theta)
        x0 = a * right_rho; y0 = b * right_rho
        offset1 = 2050; offset2 = 830
        x3 = int(x0 - offset1 * (-b)); y3 = int(y0 - offset1 * (a))
        x4 = int(x0 - offset2 * (-b)); y4 = int(y0 - offset2 * (a))

        cv2.line(snip, (x3, y3), (x4, y4), (0, 255, 0), 6)



    #to overlay lane outline on original
    if left_theta > np.pi/4 and right_theta > np.pi/4:
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)

        #create a copy of the original:
        overlay = snip.copy()
        #draw shape:
        cv2.fillConvexPoly(overlay, pts, (255, 100, 0))
        #blend with the original:
        opacity = 0.2
        cv2.addWeighted(overlay, opacity, snip, 1 - opacity, 0, snip)

    cv2.imshow("res",snip)
    #cv2.waitKey(10)


    # press the q key to break out of video
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# clear everything once finished
cap.release()
cv2.destroyAllWindows()
