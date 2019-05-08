import cv2 as cv
from src.scripts.algorithms.BaseKeypointAlgorithm import BaseKeypointAlgorithm
from timeit import default_timer as timer

class ORB(BaseKeypointAlgorithm):
    def __init__(self, parts, images, topMatches=20, drawMatches=True, iteration=None):
        super().__init__(parts, images, topMatches, drawMatches, iteration)
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.orb = cv.ORB_create()

    def calculateDescriptor(self, img):
        t = timer()
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        t = timer() - t

        return keypoints, descriptors, t


