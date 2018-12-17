from src.playground.algorithms.BaseAlgorithm import InputType
from src.playground.algorithms.ORB import ORB

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/orb/new"

orb = ORB(partType=InputType.DIRECTORY,
          parts=partsDir,
          imageType=InputType.DIRECTORY,
          images=originalDir,
          outputDir=outputDir)
orb.process()