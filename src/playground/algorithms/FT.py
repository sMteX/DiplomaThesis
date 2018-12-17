import cv2.ft as ft
import numpy as np
from src.playground.algorithms.BaseHogFT import BaseHogFT
from timeit import default_timer as timer

class FT(BaseHogFT):
    # parameter for FTransform
    kernelRadius: int
    kernel: np.ndarray

    def __init__(self, parts, images, kernelRadius=8):
        super().__init__(parts, images)
        self.kernelRadius = kernelRadius
        self.kernel = ft.createKernel(ft.LINEAR, self.kernelRadius, chn=1)

    def calculateDescriptor(self, img) -> object:
        t = timer()
        components = ft.FT02D_components(img, self.kernel)
        t = timer() - t

        return components, t

    def getSubsetParams(self, partDescriptor, imageDescriptor) -> object:
        return self.getSizeFromShape(partDescriptor.shape)[:2], \
               self.getSizeFromShape(imageDescriptor.shape)[:2], \
               (1, 1)

    def getSubset(self, descriptor, startX, startY, endX, endY) -> np.ndarray:
        return descriptor[startY:endY, startX:endX]

    def getResultPointScale(self) -> object:
        return self.kernelRadius, self.kernelRadius

    # @staticmethod
    # def getOptimalRadius(height, width):
    #     return int(np.sqrt((height * width) / 15.0))
