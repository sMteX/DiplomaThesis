from src.scripts.algorithms.BaseAlgorithm import fromDirectory
from src.scripts.algorithms.ORB import ORB

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts/300x300"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/old_single/orb/new"

orb = ORB(parts=fromDirectory(partsDir),
          images=fromDirectory(originalDir))
orb.process()
orb.writeResults(outputDir)
orb.printResults()

"""
Results:

Total time [ms]: 240.82
Average times [ms]:
    - Keypoint and descriptor computing for a part: 1.195
    - Keypoint and descriptor computing for an image: 20.375
    - Matching part with individual image: 0.694
    - Matching part with all images: 6.953
    - Processing entire part: 8.192
    
Average part descriptor size: 3288.0
Average image descriptor size: 13808.0

Deductions:
    - similar results to FAST+BRIEF (which makes sense, since ORB builds on BOTH of them and improves their qualities)
    - compared to FAST+BRIEF:
        - slower keypoint and descriptor detection 
        - faster matching of descriptors, most likely due to their much smaller size 
    - ORB seems to have larger requirements for "part" size, because it didn't even find any keypoints in 5 out of 9 cases
    - incorrectly matched the paw (parts/7.jpg) to the skyscraper image, bringing the accuracy even further down to 3/9 
    - also mismatched some of the best considered keypoints (we take top 20, yet some were wrong)

    - so far seems very bad compared to others, maybe it performs better on larger images
"""