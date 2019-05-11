import gc
import json
import os
import math
import cv2 as cv
import numpy as np
from time import strftime
from collections import namedtuple
from src.algorithms.BaseAlgorithm import InputImage
from src.algorithms.BaseAlgorithm import fromDirectory
from src.algorithms.FT import FT

dataDir = "../data"
partsDir = f"{dataDir}/parts/300x300"
originalDir = f"{dataDir}/original/300x300"
outputDir = f"{dataDir}/experimentResults/lighting"

annotations = None
with open(f"{dataDir}/parts/300x300_annotations.json", "r") as file:
    annotations = json.load(file, object_hook=lambda d: namedtuple("X", d.keys())(*d.values()))

absolute = os.path.abspath(partsDir)
paths = []
for fileName in os.listdir(absolute):
    path = os.path.abspath(f"{absolute}/{fileName}")
    paths.append(path)

formatPercentage = lambda x: f"+{x} %" if x >= 0 else f"{x} %"
formatFileName = lambda i, b, c: f"part{i}_b{b}_c{c}.jpg"

print(f"({strftime('%H:%M:%S')}) Started")
j = 0
step = 5
total = float(((160 / step) + 1)**2)
countDict = { b: {c: 0 for c in range(-80, 81, step)} for b in range(-80, 81, step)}

formatProgress = lambda x: f"{(100 * x / total):.0f} %"
for b in range(-80, 81, step):
    for c in range(-80, 81, step):
        j += 1
        print(f"({strftime('%H:%M:%S')}) ({formatProgress(j)}) Brightness: {formatPercentage(b)}, Contrast: {formatPercentage(c)}")
        images = []
        for path in paths:
            img = cv.imread(path)
            alpha = 1 + float(c)/100
            beta = float(b)/100 * 255
            image = InputImage(np.asarray(np.clip(img * alpha + beta, 0, 255), dtype=np.uint8), path)
            images.append(image)

        ft = FT(parts=images, images=fromDirectory(originalDir))
        results = ft.process()

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
            ft.writeSingleResult(result, _path, includePart=True)

        gc.collect()

print(f"({strftime('%H:%M:%S')}) Ended")
# generates a json file with the dictionary
# this needs to be manually checked and results from notSure folder that are correct have to be added into the result.json
with open(f"{outputDir}/result.json", "w") as file:
    json.dump(countDict, file)