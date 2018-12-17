import cv2 as cv
from src.playground.algorithms.BaseKeypointAlgorithm import BaseKeypointAlgorithm
from timeit import default_timer as timer

class SIFT(BaseKeypointAlgorithm):
    sift = cv.xfeatures2d.SIFT_create()

    def __init__(self, parts, images, topMatches=20, drawMatches=True):
        super().__init__(parts, images, topMatches, drawMatches)
        self.bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    def calculateDescriptor(self, img):
        t = timer()
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        t = timer() - t

        return keypoints, descriptors, t


