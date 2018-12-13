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

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/orb"

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
# SURF keypoints of the original images are the same for every processed part, we can pre-compute
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

    resultImage = cv.imread(bestResult["image"]["filename"])
    resultImage = cv.rectangle(resultImage, pt1=(startX, startY), pt2=(endX, endY), color=(0, 255, 0))
    if DRAW_MATCHES:
        resultImage = cv.drawMatches(partColor, partKeypoints, resultImage, bestResult["image"]["keypoints"],
                                     bestResult["topMatches"], resultImage,
                                     flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    print(f'Match for {filePath} found in {bestResult["image"]["filename"]}')
    cv.imwrite(os.path.abspath(f'{outputDir}/{"kp_" if DRAW_MATCHES else ""}{filename}'), resultImage)

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

Average times [ms]:
    - Keypoint and descriptor computing for a part: 0.784
    - Keypoint and descriptor computing for an image: 23.031
    - Matching an individual image with a part: 1.141
    - Sorting individual matches: 0.024
    - Matching all images to a part: 12.22
    - Processing entire part: 18.035

Average part descriptor size: 3288.0
Average image descriptor size: 13820.8

Deductions:
    - similar results to FAST+BRIEF (which makes sense, since ORB builds on BOTH of them and improves their qualities)
    - compared to FAST+BRIEF:
        - slower keypoint and descriptor detection (1.6x and 3.33x slower for part and image respectively)
        - faster matching of descriptors, most likely due to their much smaller size (60.56% and 41.4% faster)
    - ORB seems to have larger requirements for "part" size, because it didn't even find any keypoints in 5 out of 9 cases
    - incorrectly matched the paw (parts/7.jpg) to the skyscraper image, bringing the accuracy even further down to 3/9 
    - also mismatched some of the best considered keypoints (we take top 20, yet some were wrong)
    
    - so far seems very bad compared to others, maybe it performs better on larger images
"""