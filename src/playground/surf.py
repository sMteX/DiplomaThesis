from src.playground.algorithms.BaseAlgorithm import InputType
from src.playground.algorithms.SURF import SURF

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/surf/new"

surf = SURF(partType=InputType.DIRECTORY,
            parts=partsDir,
            imageType=InputType.DIRECTORY,
            images=originalDir,
            outputDir=outputDir)
surf.process()