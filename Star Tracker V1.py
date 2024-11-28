import cv2
import numpy as np
import math
import statistics
import itertools

def image_input(stars_img):
    # Setting up the frame to rotate without being cut out
    diagonal = int(math.sqrt(pow(stars_img.shape[0], 2) + pow(stars_img.shape[1], 2)))
    pad_vertical = max(0, (diagonal - stars_img.shape[0]) // 2)
    pad_horizontal = max(0, (diagonal - stars_img.shape[1]) // 2)

    frame = cv2.copyMakeBorder(stars_img, pad_vertical, pad_vertical, pad_horizontal, pad_horizontal, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 225, 300, cv2.THRESH_BINARY)

    contours, heirarchies = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    star_brightness = []
    recolor = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        M = cv2.moments(contour) # Finds the weight of the x and y coordinates
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"]) # m10 is weighted sum of x divided by m00, the area
            cY = int(M["m01"] / M["m00"]) # m01 is weighted sum of y
            cW = int(M["m10"]+M["m01"])
            centers.append((cX, cY))
            star_brightness.append(cW)

    avg = statistics.mean(star_brightness)

    bright_stars = []

    # If there are more than 100 stars, then it will filter out the stars with below average brightness
    if len(centers) > 100:
        # List of all stars that have above average brightness
        for a in range(len(centers)):
            if star_brightness[a] > avg*3:
                x,y = centers[a]
                cv2.circle(recolor, (x,y), 3, (0,0,255), 2)
                bright_stars.append((x,y))

    else:
        for x,y in centers:
            cv2.circle(recolor, (x,y), 3, (0,0,255), 2)
        bright_stars = centers

    # List of all distances between all stars
    starpairs = [tuple(x + y) for x, y in itertools.combinations(bright_stars, 2)]
    distances = []

    for i in range(len(starpairs)):
        x,y,x1,y1 = starpairs[i]
        distances.append(math.sqrt(abs((x-x1)**2)+abs((y-y1)**2)))

    # ratio between distances
    star_ratio = [x/y for x,y in itertools.permutations(distances,2)]
    cv2.imshow("", recolor)
    return [round(num, 10) for num in star_ratio]

rounded_ref = image_input(cv2.imread("orion-3.jpg"))
rounded_stars = image_input(cv2.imread("Orion-Constellation-1024x576.jpg"))
rounded_negative = image_input(cv2.imread("negative.jpg"))

# cross referencing reference with image
matches = [item for item in rounded_stars if item in rounded_ref]

print("Ref:",len(rounded_ref))
print("Orion Test:",len(rounded_stars))
print("Matches:",len(matches))
matches = [item for item in rounded_negative if item in rounded_ref]
print("Negative:",len(rounded_negative))
print("Matches:",len(matches))

cv2.waitKey(0)