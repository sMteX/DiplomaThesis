from src.playground.algorithms.BaseAlgorithm import fromDirectory
from src.playground.algorithms.HOG import HOG

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/hog/new"

hog = HOG(parts=fromDirectory(partsDir),
          images=fromDirectory(originalDir))
hog.process()
hog.writeResults(outputDir)
hog.printResults()