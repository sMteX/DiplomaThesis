from typing import List

import cv2 as cv
import os
import numpy as np
from enum import Enum
from timeit import default_timer as timer

class InputImage:
    filePath: str
    colorImage: np.ndarray

    def __init__(self, image, path=""):
        self.colorImage = image
        if path == "":
            self.filePath = "<in memory image>"
        else:
            self.filePath = path

def fromDirectory(path) -> List[InputImage]:
    absolute = os.path.abspath(path)
    result = []
    for fileName in os.listdir(absolute):
        filePath = os.path.abspath(f"{absolute}/{fileName}")
        result.append(InputImage(cv.imread(filePath), filePath))
    return result

def fromPaths(paths) -> List[InputImage]:
    result = []
    for path in os.listdir(paths):
        absolute = os.path.abspath(path)
        if not os.path.isfile(absolute):
            continue
        result.append(InputImage(cv.imread(absolute), absolute))
    return result

def fromFiles(*files) -> List[InputImage]:
    result = []
    for file in files:
        absolute = os.path.abspath(file)
        if not os.path.isfile(absolute):
            continue
        result.append(InputImage(cv.imread(absolute), absolute))
    return result

def fromImages(*images) -> List[InputImage]:
    result = []
    for image in images:
        result.append(InputImage(image))
    return result

class BaseAlgorithm:
    class Diagnostics:
        class DiagnosticTimes:
            partDescriptor = []
            imageDescriptor = []
            individualImageMatching = []
            allImagesMatching = []
            partProcess = []

        class DiagnosticCounts:
            partDescriptorSize = []
            imageDescriptorSize = []
            subsets = []

        times = DiagnosticTimes()
        counts = DiagnosticCounts()
        totalTime = -1

    class AverageType(Enum):
        TIME = 1
        COUNT = 2

    class MatchingResult:
        part: np.ndarray = None
        image: np.ndarray = None
        # rectangle for where the match was found
        start: (int, int)
        end: (int, int)
        # following variables are used in keypoint-based algorithms
        partKeypoints: List[cv.KeyPoint]
        imageKeypoints: List[cv.KeyPoint]
        topMatches: List[cv.DMatch]

        def __init__(self, part, image, start, end, partKeypoints=None, imageKeypoints=None, topMatches=None):
            self.part = part
            self.image = image
            self.start = start
            self.end = end
            self.partKeypoints = partKeypoints
            self.imageKeypoints = imageKeypoints
            self.topMatches = topMatches


    parts: List[InputImage] = []
    images: List[InputImage] = []

    diagnostics = Diagnostics()

    imageData = []
    results: List[MatchingResult] = []

    def __init__(self, parts, images):
        """
        Initializes the base matching algorithm

        :type parts: List[InputImage]
        :param parts: Images that are being matched ("parts")
        :param images: Image database to match into
        """
        self.parts = parts
        self.images = images
        self.diagnostics = self.Diagnostics()
        self.imageData = []
        self.results = []
    """
    Overall structure of the algorithm stays the same

    1) preprocess image database
       for each part:
        2) calculate descriptors
        3) somehow match them to the image database
        4) save match as result

    5) write results to files
    6) process diagnostic data
    7) print result

    - matching could be used in multiple ways
        - match 1:1 (basically check if image contains part)
        - match 1:N (matching part in a database)
        - match N:1 (find parts in image) 
        - match N:N (match all parts in database)
    - this could somehow mimic the constructor/static create method
    - for now, let's accept paths only as input
        - constructor(partType, parts, imageType, images, outputDir)
    - maybe utilize a builder pattern of some sort later
    """

    def process(self) -> List[MatchingResult]:
        self.diagnostics.totalTime = timer()
        self.processImages()
        self.processParts()
        self.diagnostics.totalTime = np.round((timer() - self.diagnostics.totalTime) * 1000, 3)
        return self.results

    # implement in child algorithms

    def processImages(self):
        pass

    def processParts(self):
        pass

    def writeResults(self, directory, includePart=False):
        pass

    def printResults(self, filename=None):
        pass

    @staticmethod
    def getSizeFromShape(shape):
        """Returns tuple (width, height) from shape (which is usually height, width)"""
        return shape[1], shape[0]

    @staticmethod
    def getCroppedSize(base, size):
        """
        Crops the size to be in multiples of base size

        :param base: Base size (width, height)
        :type base: (int, int)
        :param size: Image size (width, height)
        :type size: (int, int)
        :return: Image size cropped to be multiple of base size (width, height)
        :rtype: (int, int)
        """
        baseWidth, baseHeight = base
        width, height = size
        return width // baseWidth * baseWidth, height // baseHeight * baseHeight

    @staticmethod
    def getSubsets(windowSize, imageSize, step):
        """
        Returns an iterator generating all possible points where a part can be within an image

        :param windowSize: Size of a searched part (width, height)
        :type windowSize: (int, int)
        :param imageSize: Size of the image (width, height)
        :type imageSize: (int, int)
        :param step: Step for the generator - tuple of (stepX, stepY)
        :type step: (int, int)
        :return: Tuple (startX, startY, endX, endY) for each possible subset
        :rtype: (int, int, int, int)
        """
        pW, pH = windowSize
        iW, iH = imageSize
        stepX, stepY = step
        y = 0
        while y + pH < iH:
            x = 0
            while x + pW < iW:
                yield x, y, x + pW, y + pH
                x = x + stepX
            y = y + stepY

    @staticmethod
    def avg(array, averageType=AverageType.TIME):
        average = np.average(np.asarray(array))
        if averageType == BaseAlgorithm.AverageType.TIME:
            return np.round(average * 1000, 3)
        else:
            return np.round(average, 2)