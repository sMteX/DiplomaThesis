from src.playground.algorithms.BaseAlgorithm import InputType
from src.playground.algorithms.FREAK import FREAK

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/fast_freak/new"

freak = FREAK(partType=InputType.DIRECTORY,
              parts=partsDir,
              imageType=InputType.DIRECTORY,
              images=originalDir,
              outputDir=outputDir)
freak.process()