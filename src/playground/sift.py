from src.playground.algorithms.BaseAlgorithm import InputType
from src.playground.algorithms.SIFT import SIFT

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/sift/new"

sift = SIFT(partType=InputType.DIRECTORY,
            parts=partsDir,
            imageType=InputType.DIRECTORY,
            images=originalDir,
            outputDir=outputDir)
sift.process()