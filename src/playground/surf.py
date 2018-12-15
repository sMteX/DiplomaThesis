import cv2 as cv
import numpy as np
import os
from timeit import default_timer as timer

TOP_MATCHES = 20
DRAW_MATCHES = False

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/surf"

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

surf = cv.xfeatures2d.SURF_create()
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

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
# SURF keypoints of the original images are the same for every processed part, we can pre-compute
for i, filename in enumerate(os.listdir(originalDir)):
    filePath = os.path.abspath(f"{originalDir}/{filename}")
    img = cv.imread(filePath, 0)
    imageKeypointTime = timer()
    keypoints, descriptors = surf.detectAndCompute(img, None)
    diagnostics["computeTimes"]["imageKeypoints"].append(timer() - imageKeypointTime)
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
    partKeypoints, partDescriptors = surf.detectAndCompute(part, None)
    diagnostics["computeTimes"]["partKeypoints"].append(timer() - partKeypointTime)
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

Total time [ms]: 584.875
Average times [ms]:
    - Keypoint and descriptor computing for a part: 4.59
    - Keypoint and descriptor computing for an image: 147.789
    - Matching an individual image with a part: 1.046
    - Sorting individual matches: 0.049
    - Matching all images to a part: 12.119
    - Processing entire part: 18.276

Average part descriptor size: 2936.89
Average image descriptor size: 39628.8

Deductions:
    - almost identical implementation as SIFT
    - similar times for computing the keypoints and descriptors
    - matching is 60,91 % faster on average, processing entire part 60,71 % faster
    - this is most likely to much smaller descriptor size (and count)
    - SIFT descriptor (per keypoint) is 128 long, SURF uses size 64 descriptors
"""