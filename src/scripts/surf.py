from src.scripts.algorithms.BaseAlgorithm import fromDirectory
from src.scripts.algorithms.SURF import SURF

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts/300x300"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/old_single/surf/new"

surf = SURF(parts=fromDirectory(partsDir),
            images=fromDirectory(originalDir))
surf.process()
surf.writeResults(outputDir)
surf.printResults()

"""
Results:

Total time [ms]: 460.539
Average times [ms]:
    - Keypoint and descriptor computing for a part: 1.375
    - Keypoint and descriptor computing for an image: 40.681
    - Matching part with individual image: 0.425
    - Matching part with all images: 4.273
    - Processing entire part: 5.691
    
Average part descriptor size: 2936.89
Average image descriptor size: 39424.0

Deductions:
    - almost identical implementation as SIFT
    - similar times for computing the keypoints and descriptors
    - matching and processing entire part is a lot faster in SURF
    - this is most likely to much smaller descriptor size (and count)
    - SIFT descriptor (per keypoint) is 128 long, SURF uses size 64 descriptors
"""