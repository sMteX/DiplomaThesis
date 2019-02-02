from src.scripts.algorithms.BaseAlgorithm import fromDirectory
from src.scripts.algorithms.HOG import HOG

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts/300x300"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/old_single/hog/new"

hog = HOG(parts=fromDirectory(partsDir),
          images=fromDirectory(originalDir))
hog.process()
hog.writeResults(outputDir)
hog.printResults()

"""
Results:

Total time [ms]: 7740.505
Average times [ms]:
    - Descriptor computing for a part: 0.334
    - Descriptor computing for a image: 5.248    
    - Matching part with individual image: 85.34
    - Matching part with all images: 853.442
    - Processing entire part: 853.887

Average part descriptor size: 14464.0
Average image descriptor size: 197136.0 
Average subsets in image: 2967.11

Deductions:
    - calculating descriptors with HOG is very quick
        - improved version also allows for pre-computing the descriptors, VASTLY improving speed
    - most time is wasted on the sheer amount of image subsets for a single image
        - but AFAIK, there's no way around it, now that it's precomputed even
    - however, this is proportionate to cellSide parameter of HOG (which in return influences cellSize, blockSize and blockStride)
    - larger cellSide => smaller descriptor, less subsets => quicker iterating through all images => quicker processing 
        - might not be a problem for larger images, for small ones it's bad though
"""