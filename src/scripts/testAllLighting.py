import gc
import os
import cv2 as cv
import numpy as np
from time import strftime
from collections import namedtuple
from src.scripts.algorithms.BaseAlgorithm import fromDirectory, fromImages
from src.scripts.algorithms.HOG import HOG
from src.scripts.algorithms.FT import FT
from src.scripts.algorithms.SIFT import SIFT
from src.scripts.algorithms.SURF import SURF
from src.scripts.algorithms.BRIEF import BRIEF
from src.scripts.algorithms.ORB import ORB
from src.scripts.algorithms.FREAK import FREAK

size = "300x300"

imagesDir = "../../data/images"
partsDir = f"{imagesDir}/testing/parts/{size}"
originalDir = f"{imagesDir}/original/{size}"
outputDir = f"{imagesDir}/testing/output/lighting"

Algorithm = namedtuple("Algorithm", "name type output")

algorithms = [
    Algorithm(name="FT",    type=FT,    output=f"{outputDir}/ft"),
    # Algorithm(name="SIFT",  type=SIFT,  output=f"{outputDir}/sift"),
    # Algorithm(name="SURF",  type=SURF,  output=f"{outputDir}/surf"),
    # Algorithm(name="BRIEF", type=BRIEF, output=f"{outputDir}/fast_brief"),
    # Algorithm(name="ORB",   type=ORB,   output=f"{outputDir}/orb"),
    # Algorithm(name="FREAK", type=FREAK, output=f"{outputDir}/fast_freak"),
    # Algorithm(name="HOG",   type=HOG,   output=f"{outputDir}/hog")
]


absolute = os.path.abspath(partsDir)
paths = []
for fileName in os.listdir(absolute):
    paths.append(os.path.abspath(f"{absolute}/{fileName}"))

fn = lambda x: f"+{x} %" if x >= 0 else f"{x} %"

print(f"({strftime('%H:%M:%S')}) Started")
i = 0
for b in range(-80, 81, 20):
    for c in range(-80, 81, 20):
        i += 1
        print(f"({strftime('%H:%M:%S')}) ({(100 * i / 81.0):.0f} %) Brightness: {fn(b)}, Contrast: {fn(c)}")
        images = []
        for path in paths:
            img = cv.imread(path)
            alpha = 1 + float(c)/100
            beta = float(b)/100 * 255
            images.append(np.asarray(np.clip(img * alpha + beta, 0, 255), dtype=np.uint8))

        for a in algorithms:
            obj = a.type(parts=fromImages(*images), images=fromDirectory(originalDir))
            obj.process()
            obj.writeResults(target=lambda i: f"{a.output}/part{i}/b{b}c{c}.jpg", includePart=True)
            gc.collect()

print(f"({strftime('%H:%M:%S')}) Ended")
