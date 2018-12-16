from src.playground.algorithms.BaseAlgorithm import InputType
from src.playground.algorithms.HOG import HOG

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/hog/new"

hog = HOG(partType=InputType.DIRECTORY,
          parts=partsDir,
          imageType=InputType.DIRECTORY,
          images=originalDir,
          outputDir=outputDir)
hog.process()