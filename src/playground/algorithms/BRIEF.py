import cv2 as cv
from src.playground.algorithms.BaseKeypointAlgorithm import BaseKeypointAlgorithm
from timeit import default_timer as timer

class BRIEF(BaseKeypointAlgorithm):
    fast = cv.FastFeatureDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    def __init__(self, partType, parts, imageType, images, outputDir, topMatches=20, drawMatches=True):
        super().__init__(partType, parts, imageType, images, outputDir, topMatches, drawMatches)
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    def calculateDescriptor(self, img):
        t = timer()
        kp = self.fast.detect(img, None)
        keypoints, descriptors = self.brief.compute(img, kp)
        t = timer() - t

        return keypoints, descriptors, t


