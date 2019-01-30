import cv2 as cv
import numpy as np
import os
from timeit import default_timer as timer

TOP_MATCHES = 20
DRAW_MATCHES = True

def checkValidDetectOutput(keypoints, descriptors):
    if len(keypoints) == 0:
        return False, "No keypoints detected"
    elif descriptors is None or descriptors.size == 0:
        return False, "No descriptors computed"
    return True, ""

imagesDir = "../../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/old_single/orb"

diagnostics = {
    "computeTimes": {  # how long it takes to:
        "partKeypoints": [],  # calculate keypoints for part
        "imageKeypoints": [],  # calculate keypoints for image
        "matching": [],  # iterate through all images and aggregate matches
        "individualMatching": [],  # match individual part to individual image
        "sortingMatches": [],  # sort the individual matches
        "partProcess": [],  # process each part completely
    },
    "counts": {
        "partDescriptorSize": [],
        "imageDescriptorSize": [],
    }
}

orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

imageData = []
"""
format:
{
    filename,
    keypoints,
    descriptors
}
"""

totalTime = timer()
for i, filename in enumerate(os.listdir(originalDir)):
    filePath = os.path.abspath(f"{originalDir}/{filename}")
    img = cv.imread(filePath, 0)
    imageKeypointTime = timer()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    ok = checkValidDetectOutput(keypoints, descriptors)
    diagnostics["computeTimes"]["imageKeypoints"].append(timer() - imageKeypointTime)
    if not ok[0]:
        print(f"ERROR computing keypoints or descriptors for {filePath} ({ok[1]}), skipping...")
        continue
    diagnostics["counts"]["imageDescriptorSize"].append(descriptors.size)
    imageData.append({
        "filename": filePath,
        "keypoints": keypoints,
        "descriptors": descriptors
    })

results = []
"""
format
{
    "outputFilename"  - path of the SAVED file
    "imageFilename"  - path of the ORIGINAL file (where the match was found)
    "sX", "sY", "eX", "eY"  - coords of the rectangle

    Additional data for drawing keypoint matches

    "partFilename"  - path of the PART file
    "partKeypoints", "imageKeypoints"   - keypoints of the part and image
    "topMatches"    - selected N best matches for keypoints
}
"""

for i, filename in enumerate(os.listdir(partsDir)):
    partProcessTime = timer()
    filePath = os.path.abspath(f"{partsDir}/{filename}")
    partColor = cv.imread(filePath)
    part = cv.cvtColor(partColor, cv.COLOR_BGR2GRAY)
    partKeypointTime = timer()
    partKeypoints, partDescriptors = orb.detectAndCompute(part, None)
    diagnostics["computeTimes"]["partKeypoints"].append(timer() - partKeypointTime)
    ok = checkValidDetectOutput(partKeypoints, partDescriptors)
    if not ok[0]:
        print(f"ERROR computing keypoints or descriptors for {filePath} ({ok[1]}), skipping...")
        continue
    diagnostics["counts"]["partDescriptorSize"].append(partDescriptors.size)

    bestResult = {
        "image": None,
        "distance": float("inf"),
        "topMatches": [],
    }

    matchingTime = timer()
    for image in imageData:
        individualMatchingTime = timer()
        matches = bf.match(partDescriptors, image["descriptors"])
        diagnostics["computeTimes"]["individualMatching"].append(timer() - individualMatchingTime)

        sortingMatchesTime = timer()
        matches = sorted(matches, key=lambda x: x.distance)
        diagnostics["computeTimes"]["sortingMatches"].append(timer() - sortingMatchesTime)

        matches = matches[:TOP_MATCHES]
        # matches: List of { distance (float), queryIdx (int), trainIdx (int) }
        totalDistance = np.sum(np.asarray(list(map(lambda match: match.distance, matches))))

        if totalDistance < bestResult["distance"]:
            bestResult["image"] = image
            bestResult["distance"] = totalDistance
            bestResult["topMatches"] = matches

    end = timer()
    diagnostics["computeTimes"]["matching"].append(end - matchingTime)
    diagnostics["computeTimes"]["partProcess"].append(end - partProcessTime)

    # processing result
    bestMatch = bestResult["topMatches"][0]
    bestKeypointPart = partKeypoints[bestMatch.queryIdx]
    bestKeypointImage = bestResult["image"]["keypoints"][bestMatch.trainIdx]
    startX, startY = np.round(bestKeypointImage.pt[0] - bestKeypointPart.pt[0]).astype(int), np.round(
        bestKeypointImage.pt[1] - bestKeypointPart.pt[1]).astype(int)
    endX, endY = startX + part.shape[1], startY + part.shape[0]

    results.append({
        "outputFilename": os.path.abspath(f"{outputDir}/{'kp_' if DRAW_MATCHES else ''}{filename}"),
        "imageFilename": bestResult["image"]["filename"],
        "sX": startX,
        "sY": startY,
        "eX": endX,
        "eY": endY,
        # for DRAW_MATCHES=True
        "partFilename": filePath,
        "partKeypoints": partKeypoints,
        "imageKeypoints": bestResult["image"]["keypoints"],
        "topMatches": bestResult["topMatches"]
    })

totalTime = np.round((timer() - totalTime) * 1000, 3)

for result in results:
    resultImage = cv.imread(result["imageFilename"])
    resultImage = cv.rectangle(img=resultImage,
                               pt1=(result["sX"], result["sY"]),
                               pt2=(result["eX"], result["eY"]),
                               color=(0, 255, 0))
    if DRAW_MATCHES:
        resultImage = cv.drawMatches(img1=cv.imread(result["partFilename"]),
                                     keypoints1=result["partKeypoints"],
                                     img2=resultImage,
                                     keypoints2=result["imageKeypoints"],
                                     matches1to2=result["topMatches"],
                                     outImg=resultImage,
                                     flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(result["outputFilename"], resultImage)

average = {
    "partKeypoints": np.round(np.average(np.asarray(diagnostics["computeTimes"]["partKeypoints"])) * 1000, 3),
    "imageKeypoints": np.round(np.average(np.asarray(diagnostics["computeTimes"]["imageKeypoints"])) * 1000, 3),
    "matching": np.round(np.average(np.asarray(diagnostics["computeTimes"]["matching"])) * 1000, 3),
    "individualMatching": np.round(np.average(np.asarray(diagnostics["computeTimes"]["individualMatching"])) * 1000, 3),
    "sortingMatches": np.round(np.average(np.asarray(diagnostics["computeTimes"]["sortingMatches"])) * 1000, 3),
    "partProcess": np.round(np.average(np.asarray(diagnostics["computeTimes"]["partProcess"])) * 1000, 3),
    "partDescriptorSizes": np.round(np.average(np.asarray(diagnostics["counts"]["partDescriptorSize"])), 2),
    "imageDescriptorSizes": np.round(np.average(np.asarray(diagnostics["counts"]["imageDescriptorSize"])), 2)
}

print(f"""
Total time [ms]: {totalTime}
Average times [ms]:
- Keypoint and descriptor computing for a part: {average["partKeypoints"]}
- Keypoint and descriptor computing for an image: {average["imageKeypoints"]}
- Matching an individual image with a part: {average["individualMatching"]}
- Sorting individual matches: {average["sortingMatches"]}
- Matching all images to a part: {average["matching"]}
- Processing entire part: {average["partProcess"]}

Average part descriptor size: {average["partDescriptorSizes"]}
Average image descriptor size: {average["imageDescriptorSizes"]}
""")

# -------------------------------------

"""
Results:

Total time [ms]: 285.035
Average times [ms]:
    - Keypoint and descriptor computing for a part: 0.836
    - Keypoint and descriptor computing for an image: 21.828
    - Matching an individual image with a part: 0.791
    - Sorting individual matches: 0.022
    - Matching all images to a part: 8.603
    - Processing entire part: 10.751

Average part descriptor size: 3288.0
Average image descriptor size: 13820.8

Deductions:
    - similar results to FAST+BRIEF (which makes sense, since ORB builds on BOTH of them and improves their qualities)
    - compared to FAST+BRIEF:
        - slower keypoint and descriptor detection 
        - faster matching of descriptors, most likely due to their much smaller size 
    - ORB seems to have larger requirements for "part" size, because it didn't even find any keypoints in 5 out of 9 cases
    - incorrectly matched the paw (parts/7.jpg) to the skyscraper image, bringing the accuracy even further down to 3/9 
    - also mismatched some of the best considered keypoints (we take top 20, yet some were wrong)
    
    - so far seems very bad compared to others, maybe it performs better on larger images
"""