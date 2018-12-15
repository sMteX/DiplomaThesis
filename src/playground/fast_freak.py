import cv2 as cv
import numpy as np
import os
from timeit import default_timer as timer

TOP_MATCHES = 20
DRAW_MATCHES = False

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/fast_freak"

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

fast = cv.FastFeatureDetector_create()
freak = cv.xfeatures2d.FREAK_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

def detectAndCompute(img):
    kp = fast.detect(img, None)
    keypoints, descriptors = freak.compute(img, kp)
    result = (True, "")
    if len(kp) == 0:
        result = (False, "No keypoints detected")
    elif descriptors is None or descriptors.size == 0:
        result = (False, "No descriptors computed")
    return result, keypoints, descriptors

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
    ok, keypoints, descriptors = detectAndCompute(img)
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
    ok, partKeypoints, partDescriptors = detectAndCompute(part)
    diagnostics["computeTimes"]["partKeypoints"].append(timer() - partKeypointTime)
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

totalTime = np.round((timer() - totalTime) * 1000, 3)

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

Total time [ms]: 581.348
Average times [ms]:
    - Keypoint and descriptor computing for a part: 1.268
    - Keypoint and descriptor computing for an image: 22.372
    - Matching an individual image with a part: 4.049
    - Sorting individual matches: 0.038
    - Matching all images to a part: 41.425
    - Processing entire part: 43.654

Average part descriptor size: 9737.14
Average image descriptor size: 111052.8

Deductions:
    - uses FAST keypoint detector and FREAK descriptor
    - suffers from generally the same problem with FAST - small images
        - didn't find any descriptors in the skyscraper part and the bridge part
        - only found 1 descriptor in the cable car (part 3)?
    - generally slower than FAST + BRIEF for all parts, from descriptor computing to processing
        - most likely due to more than doubled size of descriptors
    - might be offset with better accuracy later
"""