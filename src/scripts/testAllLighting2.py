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
j = 0
step = 5
total = float(((160 / step) + 1)**2)
countDict = { b: {c: 0 for c in range(-80, 81, step)} for b in range(-80, 81, step)}
for b in range(-80, 81, step):
    for c in range(-80, 81, step):
        j += 1
        print(f"({strftime('%H:%M:%S')}) ({(100 * j / total):.0f} %) Brightness: {fn(b)}, Contrast: {fn(c)}")
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

            for i, result in enumerate(results):
                _path = None
                annotatedResult = next(x for x in annotations.matches if x.part == os.path.basename(result.partPath))
                if annotatedResult is None or os.path.basename(result.imagePath) != annotatedResult.image:
                    _path = f"{outputDir}/nope/{formatFileName(i, b, c)}"
                else:
                    _x, _y = result.start
                    distance = math.sqrt((_x - annotatedResult.x)**2 + (_y - annotatedResult.y)**2) # in "pixels"?
                    if distance < 5:
                        _path = f"{outputDir}/match/{formatFileName(i, b, c)}"
                        countDict[b][c] += 1
                    elif distance < 20:
                        _path = f"{outputDir}/notSure/{formatFileName(i, b, c)}"
                    else:
                        _path = f"{outputDir}/nope/{formatFileName(i, b, c)}"
                obj.writeSingleResult(result, _path, includePart=True)

            gc.collect()

print(f"({strftime('%H:%M:%S')}) Ended")
# generates a json file with the dictionary
# this needs to be manually checked and results from notSure folder that are correct have to be added into the result.json
with open(f"{outputDir}/result.json", "w") as file:
    json.dump(countDict, file)