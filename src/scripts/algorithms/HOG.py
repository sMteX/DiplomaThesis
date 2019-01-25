import cv2 as cv
import numpy as np
from src.scripts.algorithms.BaseHogFT import BaseHogFT
from timeit import default_timer as timer

class HOG(BaseHogFT):
    # parameters for HOGDescriptor
    cellSide: int
    cellSize: (int, int) # w x h
    blockSize: (int, int)  # w x h
    blockStride: (int, int)
    nBins = 9
    # other parameters
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nLevels = 64
    signedGradients = True

    def __init__(self, parts, images, cellSide=4):
        super().__init__(parts, images)
        self.cellSide = cellSide
        self.cellSize = (self.cellSide, self.cellSide)  # w x h
        self.blockSize = (self.cellSide * 2, self.cellSide * 2)  # w x h
        self.blockStride = self.cellSize

    def calculateDescriptor(self, img) -> object:
        croppedSize = self.getCroppedSize(self.cellSize, self.getSizeFromShape(img.shape))
        copy = cv.resize(img, croppedSize)
        hog = cv.HOGDescriptor(
            croppedSize,
            self.blockSize,
            self.blockStride,
            self.cellSize,
            self.nBins,
            self.derivAperture,
            self.winSigma,
            self.histogramNormType,
            self.L2HysThreshold,
            self.gammaCorrection,
            self.nLevels,
            self.signedGradients
        )

        t = timer()
        descriptor = hog.compute(copy)
        t = timer() - t

        newSize = self.getReshapedSize(hog.getDescriptorSize(), croppedSize, self.blockSize, self.blockStride)
        reshapedDescriptor = np.reshape(descriptor, newSize)

        return reshapedDescriptor, t

    def getSubsetParams(self, partDescriptor, imageDescriptor) -> object:
        return partDescriptor.shape[:2], imageDescriptor.shape[:2], (1, 1)

    def getSubset(self, descriptor, startX, startY, endX, endY) -> np.ndarray:
        return descriptor[startX:endX, startY:endY, :, :]

    def getResultPointScale(self) -> object:
        return self.blockStride[0], self.blockStride[1]

    @staticmethod
    def getReshapedSize(descriptorSize, winSize, blockSize, blockStride):
        w = (winSize[0] - blockSize[0]) / blockStride[0] + 1
        h = (winSize[1] - blockSize[1]) / blockStride[1] + 1
        dSize = descriptorSize / w / h
        return int(w), int(h), int(dSize), 1