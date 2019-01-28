import cv2 as cv
from src.scripts.algorithms.BaseKeypointAlgorithm import BaseKeypointAlgorithm
from timeit import default_timer as timer

class FREAK(BaseKeypointAlgorithm):
    fast = cv.FastFeatureDetector_create()
    freak = cv.xfeatures2d.FREAK_create()

    def __init__(self, parts, images, topMatches=20, drawMatches=True):
        super().__init__(parts, images, topMatches, drawMatches)
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    def calculateDescriptor(self, img):
        t = timer()
        kp = self.fast.detect(img, None)
        keypoints, descriptors = self.freak.compute(img, kp)
        t = timer() - t

        return keypoints, descriptors, t

