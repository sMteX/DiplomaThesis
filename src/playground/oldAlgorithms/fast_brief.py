import cv2 as cv
import numpy as np
import os
from timeit import default_timer as timer

TOP_MATCHES = 20
DRAW_MATCHES = True

imagesDir = "../../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/fast_brief"

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
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

def detectAndCompute(img):
    kp = fast.detect(img, None)
    keypoints, descriptors = brief.compute(img, kp)
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

Total time [ms]: 246.199
Average times [ms]:
    - Keypoint and descriptor computing for a part: 0.486
    - Keypoint and descriptor computing for an image: 6.825
    - Matching an individual image with a part: 2.392
    - Sorting individual matches: 0.025
    - Matching all images to a part: 24.609
    - Processing entire part: 25.893
    
Average part descriptor size: 4229.33
Average image descriptor size: 52163.2

Deductions:
    - uses FAST corner detector and BRIEF descriptor (neither can't be standalone)
        - FAST produces keypoints, but doesn't have compute() method for descriptors
        - BRIEF produces descriptors, but doesn't have detect() method for keypoints 
    - BRIEF uses different matching norm - Hamming distance
    - computing the descriptors is much faster than SURF
    - on the other hand, matching the part to images is much slower (but still in order of milliseconds)
    - the descriptor sizes are also larger
    - major problem is that BRIEF can't compute descriptors on small sizes 
        - BRIEF filters out keypoints in 28px range from the borders, so that makes minimum image size 57x57px
    - it also seems to sometimes compute small amount of descriptors 
        - part 4.jpg, but that's probably because of the size
        - 64x63px, after subtracting the 56x56 border, that makes it 8x7px large
        - small amount of descriptors makes sense
"""