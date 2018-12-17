from src.playground.algorithms.BaseAlgorithm import fromDirectory
from src.playground.algorithms.FT import FT

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/ft/new"

ft = FT(parts=fromDirectory(partsDir),
        images=fromDirectory(originalDir))
ft.process()
ft.writeResults(outputDir)
ft.printResults()