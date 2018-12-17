from src.playground.algorithms.BaseAlgorithm import InputType
from src.playground.algorithms.BRIEF import BRIEF

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/fast_brief/new"

brief = BRIEF(partType=InputType.DIRECTORY,
              parts=partsDir,
              imageType=InputType.DIRECTORY,
              images=originalDir,
              outputDir=outputDir)
brief.process()