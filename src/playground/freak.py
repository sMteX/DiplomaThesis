from src.playground.algorithms.BaseAlgorithm import fromDirectory
from src.playground.algorithms.FREAK import FREAK

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/fast_freak/new"

freak = FREAK(parts=fromDirectory(partsDir),
              images=fromDirectory(originalDir))
freak.process()
freak.writeResults(outputDir)
freak.printResults()

"""
Results:

Total time [ms]: 455.124
Average times [ms]:
    - Keypoint and descriptor computing for a part: 1.634
    - Keypoint and descriptor computing for an image: 21.735
    - Matching part with individual image: 3.185
    - Matching part with all images: 31.874
    - Processing entire part: 33.557
    
Average part descriptor size: 9737.14
Average image descriptor size: 110316.8

Deductions:
    - uses FAST keypoint detector and FREAK descriptor
    - suffers from generally the same problem with FAST - small images
        - didn't find any descriptors in the skyscraper part and the bridge part
        - only found 1 descriptor in the cable car (part 3)?
    - generally slower than FAST + BRIEF for all parts, from descriptor computing to processing
        - most likely due to more than doubled size of descriptors
    - might be offset with better accuracy later
"""