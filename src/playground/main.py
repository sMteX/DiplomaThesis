import cv2 as cv
import numpy as np
import json
import os
from collections import namedtuple

ORIGINAL_DIR = os.path.abspath("../../data/images/original/300x300")
PARTS_DIR = os.path.abspath("../../data/images/testing/parts/300x300")
RESULT_DIR = os.path.abspath("./test")

path = os.path.abspath("../../data/images/testing/parts/300x300_annotations.json")
with open(path, "r") as file:
    obj = json.load(file, object_hook=lambda d: namedtuple("X", d.keys())(*d.values()))

    for i, match in enumerate(obj.matches):
        part = cv.imread(f"{PARTS_DIR}/{match.part}")
        img = cv.imread(f"{ORIGINAL_DIR}/{match.image}")

        img = cv.rectangle(img,
                           pt1=(match.x, match.y),
                           pt2=(match.x + part.shape[1], match.y + part.shape[0]),
                           color=(0, 0, 255))


        result = np.zeros((img.shape[0], img.shape[1] + part.shape[1], 3), np.uint8)
        result[0:part.shape[0], 0:part.shape[1]] = part
        result[0:, part.shape[1]:] = img
        cv.imwrite(f"{RESULT_DIR}/{i}.jpg", result)