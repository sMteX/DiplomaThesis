from src.playground.algorithms.BaseAlgorithm import fromDirectory
from src.playground.algorithms.SIFT import SIFT

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/sift/new"

sift = SIFT(parts=fromDirectory(partsDir),
            images=fromDirectory(originalDir))
sift.process()
sift.writeResults(outputDir)
sift.printResults()