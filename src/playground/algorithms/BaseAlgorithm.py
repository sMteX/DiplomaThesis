import os
import numpy as np
from enum import Enum
from timeit import default_timer as timer

class InputType(Enum):
    PATH_ARRAY = 1,
    DIRECTORY = 2

class BaseAlgorithm:
    class Diagnostics:
        times = {
            "partDescriptor": [],
            "imageDescriptor": [],
            "imageProcess": [],
            "partProcess": [],
        }
        counts = {
            "partDescriptorSize": [],
            "imageDescriptorSize": [],
        }
        totalTime = -1

    class AverageType(Enum):
        TIME = 1
        COUNT = 2

    partPaths = []
    imagePaths = []
    outputDir = ""
    diagnostics = Diagnostics()

    def __init__(self, partType, parts, imageType, images, outputDir):
        """
        Initializes the base matching algorithm

        :param partType: Type of parts input (InputType enum)
        :param parts: Parts - could be either array of paths or directory
        :param imageType: Type of image input (InputType enum)
        :param images: Images - could be either array of paths or directory
        :param outputDir: Path to output directory
        """
        if partType == InputType.DIRECTORY:
            for file in os.listdir(parts):
                self.partPaths.append(os.path.abspath(f"{parts}/{file}"))
        else:
            self.partPaths = list(parts)

        if imageType == InputType.DIRECTORY:
            for file in os.listdir(images):
                self.imagePaths.append(os.path.abspath(f"{images}/{file}"))
        else:
            self.imagePaths = list(images)

        self.outputDir = outputDir

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

    def process(self):
        self.diagnostics.totalTime = timer()
        self.processImages()
        self.processParts()
        self.diagnostics.totalTime = np.round((timer() - self.diagnostics.totalTime) * 1000, 3)

        self.writeResults()
        self.printResults()

    # implement in child algorithms

    def processImages(self):
        pass

    def processParts(self):
        pass

    def writeResults(self):
        pass

    def printResults(self):
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