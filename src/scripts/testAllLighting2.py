import gc
import json
import os
import cv2 as cv
import numpy as np
from time import strftime
import math
from src.scripts.algorithms.BaseAlgorithm import InputImage
from collections import namedtuple
from src.scripts.algorithms.BaseAlgorithm import fromDirectory
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
outputDir = f"{imagesDir}/testing/output/lighting2"

annotations = None
with open(f"{imagesDir}/testing/parts/{size}_annotations.json", "r") as file:
    annotations = json.load(file, object_hook=lambda d: namedtuple("X", d.keys())(*d.values()))

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
    path = os.path.abspath(f"{absolute}/{fileName}")
    print(f"path={path}, baseName={os.path.basename(path)}")
    paths.append(path)

fn = lambda x: f"+{x} %" if x >= 0 else f"{x} %"
formatFileName = lambda i, b, c: f"part{i}_b{b}_c{c}.jpg"

print(f"({strftime('%H:%M:%S')}) Started")
i = 0
step = 40
total = float(((160 / 5) + 1)**2)
for b in range(-80, 81, step):
    for c in range(-80, 81, step):
        i += 1
        print(f"({strftime('%H:%M:%S')}) ({(100 * i / total):.0f} %) Brightness: {fn(b)}, Contrast: {fn(c)}")
        images = []
        for path in paths:
            img = cv.imread(path)
            alpha = 1 + float(c)/100
            beta = float(b)/100 * 255
            image = InputImage(np.asarray(np.clip(img * alpha + beta, 0, 255), dtype=np.uint8), path)
            images.append(image)

        for a in algorithms:
            obj = a.type(parts=images, images=fromDirectory(originalDir))
            results = obj.process()
            obj.writeResults(target=lambda i: f"{a.output}/{formatFileName(i, b, c)}", includePart=True)

            for result in results:
                annotatedResult = next(x for x in annotations.matches if x.part == os.path.basename(result.partPath))
                if annotatedResult is None or os.path.basename(result.imagePath) != annotatedResult.image:
                    continue
                _x, _y = result.start
                distance = math.sqrt((_x - annotatedResult.x)**2 + (_y - annotatedResult.y)**2) # in "pixels"?
                if distance < 5:
                    # ok
                elif distance < 20:
                    # not sure


            gc.collect()

print(f"({strftime('%H:%M:%S')}) Ended")
