import cv2 as cv
import numpy as np
import os

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

def getSubsetsFromImage(partSize, imageSize):
    """
    Returns an iterator generating all possible points where a part can be within an image
    :param partSize: Size of a searched part (width, height)
    :type partSize: (int, int)
    :param imageSize: Size of the image (width, height)
    :type imageSize: (int, int)
    :return: Tuple (startX, startY, endX, endY) for each possible subset
    :rtype: (int, int, int, int)
    """
    pW, pH = partSize
    iW, iH = imageSize
    y = 0
    while y + pH < iH:
        x = 0
        while x + pW < iW:
            yield x, y, x + pW, y + pH
            x = x + 1
        y = y + 1

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

partsCount = len(os.listdir(partsDir))
originalCount = len(os.listdir(originalDir))

for i, filename in enumerate(os.listdir(partsDir)):
    if DEBUG:
        print(f"Current file: {partsDir}/{filename}")
    # load a part
    img = cv.imread(f"{partsDir}/{filename}", 0)
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
        origImg = cv.imread(f"{originalDir}/{originalFileName}", 0)
        origSize = getSizeFromShape(origImg.shape)
        if DEBUG:
            print(f"Original width: {origSize[0]}, height: {origSize[1]}")
        for startX, startY, endX, endY in getSubsetsFromImage(croppedSize, origSize):
            subset = origImg[startY:endY, startX:endX]
            subsetSize = getSizeFromShape(subset.shape)
            if DEBUG:
                print(f"Subset: [{startX}, {startY}] -> [{endX}, {endY}]")
                print(f"Subset width: {subsetSize[0]}, height: {subsetSize[1]}")
            subsetDescriptor = hog.compute(subset)
            print(f"Subset descriptor shape: {subsetDescriptor.shape}")
            distance = np.linalg.norm(subsetDescriptor - partDescriptor)    # should calculate the euclidean distance
