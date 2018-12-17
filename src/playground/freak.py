from src.playground.algorithms.BaseAlgorithm import fromDirectory
from src.playground.algorithms.FREAK import FREAK

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/fast_freak/new"

freak = FREAK(parts=fromDirectory(partsDir),
              images=fromDirectory(originalDir))
freak.process()
freak.writeResults(outputDir)
freak.printResults()