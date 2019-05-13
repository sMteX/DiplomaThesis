from typing import List
from types import LambdaType

import cv2 as cv
import os
import numpy as np
from enum import Enum
from timeit import default_timer as timer

class InputImage:
    def __init__(self, image: np.ndarray, path: str = "") -> None:
        self.colorImage = image
        if path == "":
            self.filePath = "<in memory image>"
        else:
            self.filePath = path

# loads every image in a directory - absolute or relative
# fromDirectory("directory")
# doesn't check for file types etc.
def fromDirectory(path) -> List[InputImage]:
    absolute = os.path.abspath(path)
    result = []
    for fileName in os.listdir(absolute):
        filePath = os.path.abspath(f"{absolute}/{fileName}")
        result.append(InputImage(cv.imread(filePath), filePath))
    return result

# loads images from file paths - absolute or relative
# fromFiles("path1", "path2", ...)
# doesn't check for file types etc.
def fromFiles(*files) -> List[InputImage]:
    result = []
    for file in files:
        absolute = os.path.abspath(file)
        if not os.path.isfile(absolute):
            continue
        result.append(InputImage(cv.imread(absolute), absolute))
    return result

# loads in-memory images - OpenCV's imread etc.
def fromImages(*images) -> List[InputImage]:
    result = []
    for image in images:
        result.append(InputImage(image))
    return result

class BaseAlgorithm:
    class Diagnostics:
        class DiagnosticTimes:
            def __init__(self):
                self.partDescriptor = []
                self.imageDescriptor = []
                self.individualImageMatching = []
                self.allImagesMatching = []
                self.partProcess = []

        class DiagnosticCounts:
            def __init__(self):
                self.partDescriptorSize = []
                self.imageDescriptorSize = []
                self.subsets = []

        def __init__(self):
            self.times = self.DiagnosticTimes()
            self.counts = self.DiagnosticCounts()
            self.totalTime = -1

    class AverageType(Enum):
        TIME = 1
        COUNT = 2

    class MatchingResult:
        def __init__(self,
                     part: np.ndarray,
                     image: np.ndarray,
                     start: (int, int),
                     end: (int, int),
                     partKeypoints: List[cv.KeyPoint] = None,
                     imageKeypoints: List[cv.KeyPoint] = None,
                     topMatches: List[cv.DMatch] = None,
                     partPath: str = None,
                     imagePath: str = None) -> None:
            self.part = part
            self.image = image
            # rectangle for where the match was found
            self.start = start
            self.end = end
            self.partPath = partPath
            self.imagePath = imagePath
            # following variables are used in keypoint-based algorithms
            self.partKeypoints = partKeypoints
            self.imageKeypoints = imageKeypoints
            self.topMatches = topMatches

    def __init__(self, parts: List[InputImage], images: List[InputImage], iteration: int = None) -> None:
        """
        Initializes the base matching algorithm

        :param parts: Images that are being matched ("parts")
        :param images: Image database to match into
        """
        self.parts = parts
        self.images = images
        self.diagnostics = self.Diagnostics()
        self.imageData = []
        self.results: List[BaseAlgorithm.MatchingResult] = []
        self.iteration = iteration

    def process(self) -> List[MatchingResult]:
        # overall structure of the algorithms stays the same
        # 1) preprocess images
        # 2) process parts, which includes matching them with the images
        # (optional) 3) use results - write them into files, print diagnostic data etc.
        self.diagnostics.totalTime = timer()
        self.processImages()
        self.processParts()
        self.diagnostics.totalTime = np.round((timer() - self.diagnostics.totalTime) * 1000, 3)
        return self.results

    # implement in child algorithms

    def processImages(self):
        """
        Preprocesses loaded images
        """
        pass

    def processParts(self):
        """
        Processes loaded parts and matches them with loaded images
        """
        pass

    def writeResults(self, target, includePart=False):
        """
        Writes the match results into files

        :param target: Directory path (string) or a lambda that takes result index (int) and returns a path for the actual file
        :param includePart: Boolean, if the saved result should include the searched part or not
        """
        isLambda = isinstance(target, LambdaType)
        for i, result in enumerate(self.results):
            path = os.path.abspath(f"{target}/{i}.jpg") if not isLambda else os.path.abspath(target(i))
            self.writeSingleResult(result, path, includePart)

    def writeSingleResult(self, result, path, includePart=False):
        """
        Writes a single match result into a file

        :param result: Matching result (actual MatchingResult object)
        :param path: Path, where the result should be saved
        :param includePart: Boolean, if the saved result should include the searched part or not
        """
        pass

    def printResults(self, filename=None):
        """
        Prints the measured results into file or in the console

        :param filename: (optional) Path of the file, where the results should be saved (string)
        """
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