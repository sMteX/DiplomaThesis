import cv2 as cv
import numpy as np
import os
from timeit import default_timer as timer

DEBUG = False

def getCroppedSize(cellSize, realSize):
    """
    Crops the real image size to be in multiples of cell size

    :param cellSize: HOG cell size (width, height)
    :type cellSize: (int, int)
    :param realSize: Real image size (width, height)
    :type realSize: (int, int)
    :return: Image size cropped to be multiple of cell size (width, height)
    :rtype: (int, int)
    """
    cellWidth, cellHeight = cellSize
    width, height = realSize
    return width // cellWidth * cellWidth, height // cellHeight * cellHeight

def getSubsetsFromImage(partSize, imageSize, step):
    """
    Returns an iterator generating all possible points where a part can be within an image
    :param partSize: Size of a searched part (width, height)
    :type partSize: (int, int)
    :param imageSize: Size of the image (width, height)
    :type imageSize: (int, int)
    :param step: Step for the generator - tuple of (stepX, stepY)
    :type step: (int, int)
    :return: Tuple (startX, startY, endX, endY) for each possible subset
    :rtype: (int, int, int, int)
    """
    pW, pH = partSize
    iW, iH = imageSize
    stepX, stepY = step
    y = 0
    while y + pH < iH:
        x = 0
        while x + pW < iW:
            yield x, y, x + pW, y + pH
            x = x + stepX
        y = y + stepY

def getSizeFromShape(shape):
    """Returns tuple (width, height) from shape (which is usually height, width)"""
    return shape[1], shape[0]

# parametry pro HOGDescriptor
cellSide = 4
cellSize = (cellSide, cellSide)  # w x h
blockSize = (cellSide * 2, cellSide * 2)  # w x h
blockStride = cellSize
nBins = 9
# defaultne nastavene parametry
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nLevels = 64
signedGradients = True

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/hog"

computeTimes = {
    "partDescriptor": [],       # how long it takes for hog.compute(part)
    "subsetDescriptor": [],     # how long it takes for hog.compute(subset)
    "distanceCalculating": [],  # how long it takes to calculate the euclidean distance
    "allSubsetsCompare": [],    # how long it takes for all subsets from single original picture ("j" for loop)
    "partProcess": [],          # how long it takes to process each part ("i" for loop)
}
subsetCounts = []

for i, filename in enumerate(os.listdir(partsDir)):
    partProcessTime = timer()
    partPath = os.path.abspath(f"{partsDir}/{filename}")
    if DEBUG:
        print(f"Current file: {partPath}")
    # load a part
    img = cv.imread(partPath, 0)
    partSize = getSizeFromShape(img.shape)
    if DEBUG:
        print(f"Part width: {partSize[0]}, height: {partSize[1]}")
    # resize it to multiple of cellSize
    croppedSize = getCroppedSize(cellSize, partSize)
    if DEBUG:
        print(f"Cropped width: {croppedSize[0]}, height: {croppedSize[1]}")
    img = cv.resize(img, croppedSize)
    # construct a HOG descriptor for given size
    hog = cv.HOGDescriptor(
        croppedSize,
        blockSize,
        blockStride,
        cellSize,
        nBins,
        derivAperture,
        winSigma,
        histogramNormType,
        L2HysThreshold,
        gammaCorrection,
        nLevels,
        signedGradients
    )
    partDescriptorTime = timer()
    partDescriptor = hog.compute(img)
    computeTimes["partDescriptor"].append(timer() - partDescriptorTime)
    if DEBUG:
        print(f"Part descriptor shape: {partDescriptor.shape}")
    bestResult = {
        "distance": float("inf"),    # default to +infinity
        "file": "",
        "x": -1,
        "y": -1,
        "dX": croppedSize[0],
        "dY": croppedSize[1]
    }

    # for each part, iterate through original files and all parts of the images with the same size
    for j, originalFileName in enumerate(os.listdir(originalDir)):
        origPath = os.path.abspath(f"{originalDir}/{originalFileName}")
        origImg = cv.imread(origPath, 0)
        origSize = getSizeFromShape(origImg.shape)
        if DEBUG:
            print(f"Original width: {origSize[0]}, height: {origSize[1]}")
        allSubsetsCompareTime = timer()
        subsetCount = 0
        for startX, startY, endX, endY in getSubsetsFromImage(croppedSize, origSize, cellSize):
            subsetCount = subsetCount + 1
            subset = origImg[startY:endY, startX:endX]
            subsetSize = getSizeFromShape(subset.shape)
            if DEBUG:
                print(f"Subset: [{startX}, {startY}] -> [{endX}, {endY}]")
                print(f"Subset width: {subsetSize[0]}, height: {subsetSize[1]}")
            subsetDescriptorTime = timer()
            subsetDescriptor = hog.compute(subset)
            computeTimes["subsetDescriptor"].append(timer() - subsetDescriptorTime)
            if DEBUG:
                print(f"Subset descriptor shape: {subsetDescriptor.shape}")
            distanceCalculatingTime = timer()
            distance = np.linalg.norm(subsetDescriptor - partDescriptor)    # should calculate the euclidean distance
            computeTimes["distanceCalculating"].append(timer() - distanceCalculatingTime)
            if distance < bestResult["distance"]:
                bestResult["distance"] = distance
                bestResult["file"] = origPath
                bestResult["x"] = startX
                bestResult["y"] = startY
        subsetCounts.append(subsetCount)
        computeTimes["allSubsetsCompare"].append(timer() - allSubsetsCompareTime)

    computeTimes["partProcess"].append(timer() - partProcessTime)
    print(f"Result for {partPath} found in", bestResult)
    resultImage = cv.imread(bestResult["file"])
    resultImage = cv.rectangle(resultImage,
                               pt1=(bestResult["x"], bestResult["y"]),
                               pt2=(bestResult["x"] + bestResult["dX"], bestResult["y"] + bestResult["dY"]),
                               color=(0, 255, 0))
    cv.imwrite(os.path.abspath(f"{outputDir}/{filename}"), resultImage)

average = {
    "partDescriptor": np.round(np.average(np.asarray(computeTimes["partDescriptor"])) * 1000, 3),
    "subsetDescriptor": np.round(np.average(np.asarray(computeTimes["subsetDescriptor"])) * 1000, 3),
    "distanceCalculating": np.round(np.average(np.asarray(computeTimes["distanceCalculating"])) * 1000, 3),
    "allSubsetsCompare": np.round(np.average(np.asarray(computeTimes["allSubsetsCompare"])) * 1000, 3),
    "partProcess": np.round(np.average(np.asarray(computeTimes["partProcess"])) * 1000, 3)
}
averageSubsets = np.round(np.average(np.asarray(subsetCounts)), 2)

print(f"""
Average times [ms]:
- Descriptor computing for a part: {average["partDescriptor"]}
- Descriptor computing for a subset: {average["subsetDescriptor"]}
- Calculating distance between descriptors: {average["distanceCalculating"]}
- Processing all subsets (average {averageSubsets} subsets) for a single image: {average["allSubsetsCompare"]}
- Processing entire part: {average["partProcess"]}
""")

# -----------------------------------------------------------------------------------

"""
Results:

Average times [ms]:
    - Descriptor computing for a part: 0.351
    - Descriptor computing for a subset: 0.238
    - Calculating distance between descriptors: 0.023
    - Processing all subsets (average 2967.11 subsets) for a single image: 793.992
    - Processing entire part: 7959.978
    
Deductions:
    - calculating descriptors with HOG is very quick
    - calculating Euclidean distances is quick too, even for large vectors
    - most time is wasted on the sheer amount of image subsets for a single image
    - resulting in times    `number_of_parts * number_of_images * subsets_per_image * negligible_compute_time_per_subset`
"""
