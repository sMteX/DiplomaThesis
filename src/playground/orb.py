from src.playground.algorithms.BaseAlgorithm import fromDirectory
from src.playground.algorithms.ORB import ORB

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/orb/new"

orb = ORB(parts=fromDirectory(partsDir),
          images=fromDirectory(originalDir))
orb.process()
orb.writeResults(outputDir)
orb.printResults()