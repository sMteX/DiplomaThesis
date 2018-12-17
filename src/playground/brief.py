from src.playground.algorithms.BaseAlgorithm import fromDirectory
from src.playground.algorithms.BRIEF import BRIEF

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/fast_brief/new"

brief = BRIEF(parts=fromDirectory(partsDir),
              images=fromDirectory(originalDir))
brief.process()
brief.writeResults(outputDir)
brief.printResults()