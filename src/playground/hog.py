import cv2 as cv
import numpy as np
import os
from enum import Enum
from timeit import default_timer as timer

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

def getReshapedSize(descriptorSize, winSize, blockSize, blockStride):
    w = (winSize[0] - blockSize[0]) / blockStride[0] + 1
    h = (winSize[1] - blockSize[1]) / blockStride[1] + 1
    dSize = descriptorSize / w / h
    return int(w), int(h), int(dSize), 1

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

imageData = []
"""
format
{
    "descriptors": reshaped array
    "filename": path
}
"""

diagnostics = {
    "times": {
        "partDescriptor": [],
        "imageDescriptor": [],
        "distanceComputing": [],
        "imageProcess": [],
        "partProcess": [],
    },
    "counts": {
        "partDescriptorSize": [],
        "imageDescriptorSize": [],
        "subsets": [],
    }
}

totalTime = timer()

for i, filename in enumerate(os.listdir(originalDir)):
    filePath = os.path.abspath(f"{originalDir}/{filename}")
    img = cv.imread(filePath, 0)
    croppedSize = getCroppedSize(cellSize, getSizeFromShape(img.shape))
    img = cv.resize(img, croppedSize)
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

    imageDescriptorTime = timer()
    descriptor = hog.compute(img)
    diagnostics["times"]["imageDescriptor"].append(timer() - imageDescriptorTime)

    newSize = getReshapedSize(hog.getDescriptorSize(), croppedSize, blockSize, blockStride)
    reshapedDescriptor = np.reshape(descriptor, newSize)

    diagnostics["counts"]["imageDescriptorSize"].append(reshapedDescriptor.size)

    imageData.append({
        "filename": filePath,
        "descriptors": reshapedDescriptor
    })

results = []    # in order to not count drawing and saving images into total time, refactor out
"""
format
{
    "filename"  - path of the SAVED file
    "original"  - path of the ORIGINAL file (where the match was found)
    "sX", "sY", "eX", "eY"  - coords of the rectangle
}
"""

for i, filename in enumerate(os.listdir(partsDir)):
    partProcessTime = timer()
    partPath = os.path.abspath(f"{partsDir}/{filename}")
    img = cv.imread(partPath, 0)
    partSize = getSizeFromShape(img.shape)
    # resize it to multiple of cellSize
    croppedSize = getCroppedSize(cellSize, partSize)
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
    originalPartDescriptor = hog.compute(img)
    diagnostics["times"]["partDescriptor"].append(timer() - partDescriptorTime)

    newPartSize = getReshapedSize(hog.getDescriptorSize(), croppedSize, blockSize, blockStride)
    partDescriptor = np.reshape(originalPartDescriptor, newPartSize)

    diagnostics["counts"]["partDescriptorSize"].append(partDescriptor.size)

    best = {
        "distance": float("inf"),    # default to +infinity
        "filename": "",
        "sX": -1,
        "sY": -1,
        "eX": -1,
        "eY": -1
    }

    # for each part, iterate through original files and all parts of the images with the same size
    for image in imageData:
        imageProcessTime = timer()
        subsets = 0
        for startX, startY, endX, endY in getSubsetsFromImage(partDescriptor.shape[:2], image["descriptors"].shape[:2], (1, 1)):
            subsets = subsets + 1
            subset = image["descriptors"][startX:endX, startY:endY, :, :]

            distanceTime = timer()
            distance = np.linalg.norm(subset - partDescriptor)    # should calculate the euclidean distance
            diagnostics["times"]["distanceComputing"].append(timer() - distanceTime)

            if distance < best["distance"]:
                best["distance"] = distance
                best["filename"] = image["filename"]
                best["sX"] = startX * blockStride[0]
                best["sY"] = startY * blockStride[1]
                best["eX"] = best["sX"] + partSize[0]
                best["eY"] = best["sY"] + partSize[1]
        diagnostics["times"]["imageProcess"].append(timer() - imageProcessTime)
        diagnostics["counts"]["subsets"].append(subsets)

    diagnostics["times"]["partProcess"].append(timer() - partProcessTime)

    results.append({
        "filename": os.path.abspath(f"{outputDir}/new_{filename}"),
        "original": best["filename"],
        "sX": best["sX"],
        "sY": best["sY"],
        "eX": best["eX"],
        "eY": best["eY"]
    })

totalTime = np.round((timer() - totalTime) * 1000, 3)

for result in results:
    resultImage = cv.imread(result["original"])
    resultImage = cv.rectangle(resultImage,
                               pt1=(result["sX"], result["sY"]),
                               pt2=(result["eX"], result["eY"]),
                               color=(0, 0, 255))
    cv.imwrite(result["filename"], resultImage)

class AverageType(Enum):
    TIME = 1
    COUNT = 2

def avg(array, averageType=AverageType.TIME):
    average = np.average(np.asarray(array))
    if averageType == AverageType.TIME:
        return np.round(average * 1000, 3)
    else:
        return np.round(average, 2)

average = {
    "partDescriptor": avg(diagnostics["times"]["partDescriptor"]),
    "imageDescriptor": avg(diagnostics["times"]["imageDescriptor"]),
    "distanceComputing": avg(diagnostics["times"]["distanceComputing"]),
    "imageProcess": avg(diagnostics["times"]["imageProcess"]),
    "partProcess": avg(diagnostics["times"]["partProcess"]),

    "partDescriptorSize": avg(diagnostics["counts"]["partDescriptorSize"], AverageType.COUNT),
    "imageDescriptorSize": avg(diagnostics["counts"]["imageDescriptorSize"], AverageType.COUNT),
    "subsets": avg(diagnostics["counts"]["subsets"], AverageType.COUNT),
}

print(f"""
Total time [ms]: {totalTime}
Average times [ms]:
- Descriptor computing for a part: {average["partDescriptor"]}
- Descriptor computing for a image: {average["imageDescriptor"]}
- Calculating distance between descriptors: {average["distanceComputing"]}
- Processing all subsets (average {average["subsets"]} subsets) for a single image: {average["imageProcess"]}
- Processing entire part: {average["partProcess"]}

Average part descriptor size: {average["partDescriptorSize"]}
Average image descriptor size: {average["imageDescriptorSize"]} 
""")

# -----------------------------------------------------------------------------------

"""
Results before rework:

Average times [ms]:
    - Descriptor computing for a part: 0.351
    - Descriptor computing for a subset: 0.238
    - Calculating distance between descriptors: 0.023
    - Processing all subsets (average 2967.11 subsets) for a single image: 793.992
    - Processing entire part: 7959.978
    
Average descriptor size: 14464.0 numbers    
    
    
Results after rework:

Total time [ms]: 8762.391
Average times [ms]:
    - Descriptor computing for a part: 0.333
    - Descriptor computing for a image: 5.399
    - Calculating distance between descriptors: 0.027
    - Processing all subsets (average 2967.11 subsets) for a single image: 91.028
    - Processing entire part: 911.149
    
Average part descriptor size: 14464.0
Average image descriptor size: 197136.0 

Deductions:
    - calculating descriptors with HOG is very quick
        - improved version also allows for pre-computing the descriptors, VASTLY improving speed (over 8.7x faster)
    - most time is wasted on the sheer amount of image subsets for a single image
        - but AFAIK, there's no way around it, now that it's precomputed even
    - however, this is proportionate to cellSide parameter of HOG (which in return influences cellSize, blockSize and blockStride)
    - larger cellSide => smaller descriptor, less subsets => quicker iterating through all images => quicker processing 
        - might not be a problem for larger images, for small ones it's bad though
"""
