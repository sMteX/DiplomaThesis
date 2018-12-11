import cv2 as cv
import numpy as np
import os
from timeit import default_timer as timer

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/sift"

sift = cv.xfeatures2d.SIFT_create()
partColor = cv.imread(f"{partsDir}/1.jpg")
imageColor = cv.imread(f"{originalDir}/picsum.photos.jpg")
part = cv.cvtColor(partColor, cv.COLOR_BGR2GRAY)
image = cv.cvtColor(imageColor, cv.COLOR_BGR2GRAY)
kp, des = sift.detectAndCompute(part, None)
kp2, des2 = sift.detectAndCompute(image, None)

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des, des2)
matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.
bestMatch = matches[0]
bmKp1 = kp[bestMatch.queryIdx]
bmKp2 = kp2[bestMatch.trainIdx]
startX, startY = np.round(bmKp2.pt[0] - bmKp1.pt[0]).astype(int), np.round(bmKp2.pt[1] - bmKp1.pt[1]).astype(int)
endX, endY = startX + part.shape[1], startY + part.shape[0]
img_matches = cv.rectangle(imageColor, pt1=(startX, startY), pt2=(endX, endY), color=(0, 255, 0))
img_matches = cv.drawMatches(partColor, kp, img_matches, kp2, matches[:1], img_matches, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv.imwrite(os.path.abspath(f"{outputDir}/test_pair.jpg"), img_matches)
"""
For part 1.jpg (131x163px):
- 295 keypoint
- each keypoint has:
    - angle (float)
    - class_id (int)
    - octave (int)
    - pt (tuple (int, int))
    - response (float)
    - size (float)
=> descriptor size: 2065 numbers (for the largest of all available parts) 
= average will be smaller, which already is about 90% smaller than average HOG descriptor
"""