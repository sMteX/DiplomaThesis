from src.playground.algorithms.BaseAlgorithm import fromDirectory
from src.playground.algorithms.BRIEF import BRIEF

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/fast_brief/new"

brief = BRIEF(parts=fromDirectory(partsDir),
              images=fromDirectory(originalDir))
brief.process()
brief.writeResults(outputDir)
brief.printResults()

"""
Results:

Total time [ms]: 230.885
Average times [ms]:
    - Keypoint and descriptor computing for a part: 0.698
    - Keypoint and descriptor computing for an image: 7.024
    - Matching part with individual image: 2.554
    - Matching part with all images: 25.561
    - Processing entire part: 26.311
    
Average part descriptor size: 4229.33
Average image descriptor size: 51843.2

Deductions:
    - uses FAST corner detector and BRIEF descriptor (neither can't be standalone)
        - FAST produces keypoints, but doesn't have compute() method for descriptors
        - BRIEF produces descriptors, but doesn't have detect() method for keypoints 
    - BRIEF uses different matching norm - Hamming distance
    - computing the descriptors is much faster than SURF
    - on the other hand, matching the part to images is much slower (but still in order of milliseconds)
    - the descriptor sizes are also larger
    - major problem is that BRIEF can't compute descriptors on small sizes 
        - BRIEF filters out keypoints in 28px range from the borders, so that makes minimum image size 57x57px
    - it also seems to sometimes compute small amount of descriptors 
        - part 4.jpg, but that's probably because of the size
        - 64x63px, after subtracting the 56x56 border, that makes it 8x7px large
        - small amount of descriptors makes sense
"""