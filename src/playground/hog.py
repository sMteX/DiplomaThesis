import cv2 as cv
import numpy as np
import os
from timeit import default_timer as timer   # TODO

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

for i, filename in enumerate(os.listdir(partsDir)):
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
    partDescriptor = hog.compute(img)
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
        for startX, startY, endX, endY in getSubsetsFromImage(croppedSize, origSize, cellSize):
            subset = origImg[startY:endY, startX:endX]
            subsetSize = getSizeFromShape(subset.shape)
            if DEBUG:
                print(f"Subset: [{startX}, {startY}] -> [{endX}, {endY}]")
                print(f"Subset width: {subsetSize[0]}, height: {subsetSize[1]}")
            subsetDescriptor = hog.compute(subset)
            if DEBUG:
                print(f"Subset descriptor shape: {subsetDescriptor.shape}")
            distance = np.linalg.norm(subsetDescriptor - partDescriptor)    # should calculate the euclidean distance
            if distance < bestResult["distance"]:
                bestResult["distance"] = distance
                bestResult["file"] = origPath
                bestResult["x"] = startX
                bestResult["y"] = startY

    print(f"Result for {partPath} found in", bestResult)
    resultImage = cv.imread(bestResult["file"])
    resultImage = cv.rectangle(resultImage,
                               pt1=(bestResult["x"], bestResult["y"]),
                               pt2=(bestResult["x"] + bestResult["dX"], bestResult["y"] + bestResult["dY"]),
                               color=(0, 255, 0))
    cv.imwrite(os.path.abspath(f"{outputDir}/{filename}"), resultImage)