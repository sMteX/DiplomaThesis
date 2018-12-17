from src.playground.algorithms.BaseAlgorithm import fromDirectory
from src.playground.algorithms.SURF import SURF

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/surf/new"

surf = SURF(parts=fromDirectory(partsDir),
            images=fromDirectory(originalDir))
surf.process()
surf.writeResults(outputDir)
surf.printResults()