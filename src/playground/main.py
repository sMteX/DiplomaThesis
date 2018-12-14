import cv2 as cv
import numpy as np
import os

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

def getSizeFromShape(shape):
    """Returns tuple (width, height) from shape (which is usually height, width)"""
    return shape[1], shape[0]

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

partPath = os.path.abspath(f"{partsDir}/2.jpg")
originalPart = cv.imread(partPath, 0)
partSize = getSizeFromShape(originalPart.shape)
croppedSize = getCroppedSize(cellSize, partSize)
part = cv.resize(originalPart, croppedSize)
hogPart = cv.HOGDescriptor(
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
partDescriptor = hogPart.compute(part)

print(f"""
Original image size: {partSize[0]}w x {partSize[1]}h
Cropped size: {croppedSize[0]}w x {croppedSize[1]}h
Cell side: {cellSide}
Cell size: {cellSize}
Block size: {blockSize}
Block stride: {blockStride}
Descriptor size (method): {hogPart.getDescriptorSize()}
Descriptor shape: {partDescriptor.shape}
""")

imagePath = os.path.abspath(f"{originalDir}/picsum.photos1.jpg")
originalImage = cv.imread(imagePath, 0)
imageSize = getSizeFromShape(originalImage.shape)
croppedImageSize = getCroppedSize(cellSize, imageSize)
original = cv.resize(originalImage, croppedImageSize)
hogOriginal = cv.HOGDescriptor(
    croppedImageSize,
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
imageDescriptor = hogOriginal.compute(originalImage)

print(f"""
Original image size: {imageSize[0]}w x {imageSize[1]}h
Cropped size: {croppedImageSize[0]}w x {croppedImageSize[1]}h
Cell side: {cellSide}
Cell size: {cellSize}
Block size: {blockSize}
Block stride: {blockStride}
Descriptor size (method): {hogOriginal.getDescriptorSize()}
Descriptor shape: {imageDescriptor.shape}
""")

# descriptorSize / ((winSizeW - blockSizeW) / blockStrideW + 1) / ((winSizeH - blockSizeH) / blockStrideH + 1)  (for W and H) should give descriptor size per normalized block?

def getDescriptorSizePerBlock(size, winSize, blockSize, blockStride):
    return size / ((winSize[0] - blockSize[0]) / blockStride[0] + 1) / ((winSize[1] - blockSize[1]) / blockStride[1] + 1)

def getBlockCount(size, winSize, blockSize, blockStride):
    descSize = getDescriptorSizePerBlock(size, winSize, blockSize, blockStride)
    w = (winSize[0] - blockSize[0]) / blockStride[0] + 1
    h = (winSize[1] - blockSize[1]) / blockStride[1] + 1
    assert size == descSize * w * h
    return w, h

def getReshapedSize(size, winSize, blockSize, blockStride):
    w = (winSize[0] - blockSize[0]) / blockStride[0] + 1
    h = (winSize[1] - blockSize[1]) / blockStride[1] + 1
    dSize = size / w / h
    return int(w), int(h), int(dSize), 1


reshapedPartSize = getReshapedSize(hogPart.getDescriptorSize(), croppedSize, blockSize, blockStride)
reshapedPartDescriptor = np.reshape(partDescriptor, reshapedPartSize)
reshapedImageSize = getReshapedSize(hogOriginal.getDescriptorSize(), croppedImageSize, blockSize, blockStride)
reshapedImageDescriptor = np.reshape(imageDescriptor, reshapedImageSize)

bestResult = {
    "distance": float("inf"),
    "sX": -1,
    "sY": -1,
    "eX": -1,
    "eY": -1,
}

for startX, startY, endX, endY in getSubsetsFromImage(reshapedPartSize[:2], reshapedImageSize[:2], (1, 1)):
    subset = reshapedImageDescriptor[startX:endX, startY:endY, :, :]

    distance = np.linalg.norm(subset - reshapedPartDescriptor)  # should calculate the euclidean distance
    if distance < bestResult["distance"]:
        bestResult["distance"] = distance
        bestResult["sX"] = startX * blockStride[0]
        bestResult["sY"] = startY * blockStride[1]
        bestResult["eX"] = bestResult["sX"] + partSize[0]
        bestResult["eY"] = bestResult["sY"] + partSize[1]


print(bestResult)
result = cv.rectangle(img=cv.imread(imagePath), pt1=(bestResult["sX"], bestResult["sY"]), pt2=(bestResult["eX"], bestResult["eY"]), color=(0, 0, 255))
cv.imwrite(os.path.abspath(f"{outputDir}/testNew.jpg"), result)