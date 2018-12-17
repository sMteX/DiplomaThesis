import cv2 as cv
from src.playground.algorithms.BaseKeypointAlgorithm import BaseKeypointAlgorithm
from timeit import default_timer as timer

class SURF(BaseKeypointAlgorithm):
    surf = cv.xfeatures2d.SURF_create()

    def __init__(self, partType, parts, imageType, images, outputDir, topMatches=20, drawMatches=True):
        super().__init__(partType, parts, imageType, images, outputDir, topMatches, drawMatches)
        self.bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    def calculateDescriptor(self, img):
        t = timer()
        keypoints, descriptors = self.surf.detectAndCompute(img, None)
        t = timer() - t

        return keypoints, descriptors, t


