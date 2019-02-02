import gc
from time import strftime
from collections import namedtuple
from src.scripts.algorithms.BaseAlgorithm import fromDirectory
from src.scripts.algorithms.HOG import HOG
from src.scripts.algorithms.FT import FT
from src.scripts.algorithms.SIFT import SIFT
from src.scripts.algorithms.SURF import SURF
from src.scripts.algorithms.BRIEF import BRIEF
from src.scripts.algorithms.ORB import ORB
from src.scripts.algorithms.FREAK import FREAK

imagesDir = "../../data/images"
partsDir = f"{imagesDir}/testing/parts/300x300"
originalDir = f"{imagesDir}/original/300x300"
outputDir = f"{imagesDir}/testing/output/300x300"

Algorithm = namedtuple("Algorithm", "name type output")

algorithms = [
    Algorithm(name="FT",    type=FT,    output=f"{outputDir}/ft"),
    Algorithm(name="SIFT",  type=SIFT,  output=f"{outputDir}/sift"),
    Algorithm(name="SURF",  type=SURF,  output=f"{outputDir}/surf"),
    Algorithm(name="BRIEF", type=BRIEF, output=f"{outputDir}/fast_brief"),
    Algorithm(name="ORB",   type=ORB,   output=f"{outputDir}/orb"),
    Algorithm(name="FREAK", type=FREAK, output=f"{outputDir}/fast_freak"),
    Algorithm(name="HOG",   type=HOG,   output=f"{outputDir}/hog")
]

print(f"({strftime('%H:%M:%S')}) Started")

for a in algorithms:
    print(f"({strftime('%H:%M:%S')}) Algorithm: {a.name}")
    if a.name == "HOG":
        print("Get yourself a cup of coffee, this will take almost an hour")
    for i in range(10):
        print(f"({strftime('%H:%M:%S')}) - Iteration {i + 1}")
        obj = a.type(parts=fromDirectory(partsDir), images=fromDirectory(originalDir))
        obj.process()
        obj.writeResults(f"{a.output}/{i}", includePart=True)
        obj.printResults(f"{a.output}/{i}_result.txt")
        gc.collect()

print(f"({strftime('%H:%M:%S')}) Ended")
