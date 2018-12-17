from src.playground.algorithms.BaseAlgorithm import InputType
from src.playground.algorithms.FT import FT

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/ft/new"

ft = FT(partType=InputType.DIRECTORY,
        parts=partsDir,
        imageType=InputType.DIRECTORY,
        images=originalDir,
        outputDir=outputDir,
        kernelRadius=8)
ft.process()