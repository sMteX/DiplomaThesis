from src.algorithms.BaseAlgorithm import fromDirectory
from src.algorithms.SIFT import SIFT

dataDir = "../../data"
partsDir = f"{dataDir}/parts/300x300"
originalDir = f"{dataDir}/original/300x300"
outputDir = f"{dataDir}/experimentResults/old_single/sift/new"

sift = SIFT(parts=fromDirectory(partsDir),
            images=fromDirectory(originalDir))
sift.process()
sift.writeResults(outputDir)
sift.printResults()

"""
Results:

Total time [ms]: 618.98
Average times [ms]:
    - Keypoint and descriptor computing for a part: 4.767
    - Keypoint and descriptor computing for an image: 47.002
    - Matching part with individual image: 1.141
    - Matching part with all images: 11.429
    - Processing entire part: 16.254

Average part descriptor size: 12458.67
Average image descriptor size: 74598.4

Deductions:
    - compared to HOG, SIFT is much faster and I think it should be more robust 
    - after improving HOG, both pre-compute their descriptors, so the difference is most likely in the descriptor sizes
    - average image descriptor for SIFT is 62.16 % smaller than HOG = that might be responsible for the 97.86 % speed increase over HOG 
"""