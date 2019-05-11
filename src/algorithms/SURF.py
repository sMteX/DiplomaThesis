import cv2 as cv
from src.algorithms.BaseKeypointAlgorithm import BaseKeypointAlgorithm
from timeit import default_timer as timer

class SURF(BaseKeypointAlgorithm):
    def __init__(self, parts, images, topMatches=20, drawMatches=True, iteration=None):
        super().__init__(parts, images, topMatches, drawMatches, iteration)
        self.bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        self.surf = cv.xfeatures2d.SURF_create()

    def calculateDescriptor(self, img):
        t = timer()
        keypoints, descriptors = self.surf.detectAndCompute(img, None)
        t = timer() - t

        return keypoints, descriptors, t


