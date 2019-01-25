import cv2 as cv
import cv2.ft as ft
import numpy as np
import os
from enum import Enum
from timeit import default_timer as timer


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

def getOptimalRadius(height, width):
    return int(np.sqrt((height * width) / 15.0))

kernelRadius = 8

imagesDir = "../../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/ft"

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
        "partComponents": [],
        "imageComponents": [],
        "distanceComputing": [],
        "imageProcess": [],
        "partProcess": [],
    },
    "counts": {
        "partComponentSize": [],
        "imageComponentSize": [],
        "subsets": [],
    }
}

totalTime = timer()

kernel = ft.createKernel(ft.LINEAR, kernelRadius, chn=1)

for i, filename in enumerate(os.listdir(originalDir)):
    filePath = os.path.abspath(f"{originalDir}/{filename}")
    img = cv.imread(filePath, 0)

    imageComponentTime = timer()
    components = ft.FT02D_components(img, kernel)
    diagnostics["times"]["imageComponents"].append(timer() - imageComponentTime)

    diagnostics["counts"]["imageComponentSize"].append(components.size)

    imageData.append({
        "filename": filePath,
        "descriptors": components
    })

results = []
"""
format
{
    "outputFilename"  - path of the SAVED file
    "imageFilename"  - path of the ORIGINAL file (where the match was found)
    "sX", "sY", "eX", "eY"  - coords of the rectangle
}
"""

for i, filename in enumerate(os.listdir(partsDir)):
    partProcessTime = timer()
    partPath = os.path.abspath(f"{partsDir}/{filename}")
    img = cv.imread(partPath, 0)
    partSize = getSizeFromShape(img.shape)

    partComponentTime = timer()
    partComponents = ft.FT02D_components(img, kernel)
    diagnostics["times"]["partComponents"].append(timer() - partComponentTime)

    diagnostics["counts"]["partComponentSize"].append(partComponents.size)

    best = {
        "distance": float("inf"),  # default to +infinity
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
        for startX, startY, endX, endY in getSubsetsFromImage(getSizeFromShape(partComponents.shape)[:2],
                                                              getSizeFromShape(image["descriptors"].shape)[:2],
                                                              (1, 1)):
            subsets = subsets + 1
            subset = image["descriptors"][startY:endY, startX:endX]

            distanceTime = timer()
            distance = np.linalg.norm(subset - partComponents)  # should calculate the euclidean distance
            diagnostics["times"]["distanceComputing"].append(timer() - distanceTime)

            if distance < best["distance"]:
                best["distance"] = distance
                best["filename"] = image["filename"]
                best["sX"] = startX * kernelRadius
                best["sY"] = startY * kernelRadius
                best["eX"] = best["sX"] + partSize[0]
                best["eY"] = best["sY"] + partSize[1]

        diagnostics["times"]["imageProcess"].append(timer() - imageProcessTime)
        diagnostics["counts"]["subsets"].append(subsets)

    diagnostics["times"]["partProcess"].append(timer() - partProcessTime)

    results.append({
        "outputFilename": os.path.abspath(f"{outputDir}/{filename}"),
        "imageFilename": best["filename"],
        "sX": best["sX"],
        "sY": best["sY"],
        "eX": best["eX"],
        "eY": best["eY"]
    })

totalTime = np.round((timer() - totalTime) * 1000, 3)

for result in results:
    resultImage = cv.imread(result["imageFilename"])
    resultImage = cv.rectangle(resultImage,
                               pt1=(result["sX"], result["sY"]),
                               pt2=(result["eX"], result["eY"]),
                               color=(0, 0, 255))
    cv.imwrite(result["outputFilename"], resultImage)

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
    "partComponents": avg(diagnostics["times"]["partComponents"]),
    "imageComponents": avg(diagnostics["times"]["imageComponents"]),
    "distanceComputing": avg(diagnostics["times"]["distanceComputing"]),
    "imageProcess": avg(diagnostics["times"]["imageProcess"]),
    "partProcess": avg(diagnostics["times"]["partProcess"]),

    "partComponentSize": avg(diagnostics["counts"]["partComponentSize"], AverageType.COUNT),
    "imageComponentSize": avg(diagnostics["counts"]["imageComponentSize"], AverageType.COUNT),
    "subsets": avg(diagnostics["counts"]["subsets"], AverageType.COUNT),
}

print(f"""
Total time [ms]: {totalTime}
Average times [ms]:
- Component computing for a part: {average["partComponents"]}
- Component computing for an image: {average["imageComponents"]}
- Calculating distance between components: {average["distanceComputing"]}
- Processing all subsets (average {average["subsets"]} subsets) for a single image: {average["imageProcess"]}
- Processing entire part: {average["partProcess"]}

Average part component size: {average["partComponentSize"]}
Average image component size: {average["imageComponentSize"]} 
""")

# -----------------------------------------------------------------------------------

"""
Results:

Total time [ms]: 1155.361
Average times [ms]:
    - Component computing for a part: 0.768
    - Component computing for an image: 7.646
    - Calculating distance between components: 0.013
    - Processing all subsets (average 727.67 subsets) for a single image: 11.629
    - Processing entire part: 117.62

Average part component size: 128.11
Average image component size: 1444.0  

Deductions:
    - comparing the results to HOG, since both of them sort of consider the picture as basically one giant keypoint:
        - FT is much faster, about 87.49 % faster for processing entire part
        - this is most likely to drastically smaller descriptor size and subset count 
            - nearly 113x smaller part descriptor size 
            - about 136x smaller image descriptor size
            - also about 4x less subsets for a single image
            => all means less comparing than HOG
    - speed and accuracy is controlled by kernelRadius
    - main drawback of this will be most likely the fact, that FT doesn't really take into account any structure of the image
        - it's directly using pixel intensities and nothing else
        - I assume it'll perform badly on any serious distortions
"""
