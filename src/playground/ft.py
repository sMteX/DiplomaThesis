from src.playground.algorithms.BaseAlgorithm import fromDirectory
from src.playground.algorithms.FT import FT

imagesDir = "../../data/images"
partsDir = imagesDir + "/testing/parts"
originalDir = imagesDir + "/original/300x300"
outputDir = imagesDir + "/testing/output/ft/new"

ft = FT(parts=fromDirectory(partsDir),
        images=fromDirectory(originalDir))
ft.process()
ft.writeResults(outputDir)
ft.printResults()

"""
Results:

Total time [ms]: 872.91
Average times [ms]:
    - Descriptor computing for a part: 0.673
    - Descriptor computing for a image: 7.644
    - Matching part with individual image: 8.757
    - Matching part with all images: 87.593
    - Processing entire part: 88.319
    
Average part descriptor size: 128.11
Average image descriptor size: 1444.0
Average subsets in image: 727.67

Deductions:
    - comparing the results to HOG, since both of them sort of consider the picture as basically one giant keypoint:
        - FT is much faster
        - this is most likely to drastically smaller descriptor size and subset count 
            - nearly 113x smaller part descriptor size 
            - about 136x smaller image descriptor size
            - also about 4x less subsets for a single image
            => all means less comparing than HOG
    - speed and accuracy is controlled by kernelRadius
    - main drawback of this will be most likely the fact, that FT doesn't really take into account any abstract structure of the image
        - it's directly using pixel intensities and nothing else
        - I assume it'll perform badly on any serious distortions
"""