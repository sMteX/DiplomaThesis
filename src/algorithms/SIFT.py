import cv2 as cv
from src.algorithms.BaseKeypointAlgorithm import BaseKeypointAlgorithm
from timeit import default_timer as timer

class SIFT(BaseKeypointAlgorithm):
    def __init__(self, parts, images, topMatches=20, drawMatches=True, iteration=None):
        super().__init__(parts, images, topMatches, drawMatches, iteration)
        self.bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        self.sift = cv.xfeatures2d.SIFT_create()

    def calculateDescriptor(self, img):
        t = timer()
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        t = timer() - t

        return keypoints, descriptors, t


