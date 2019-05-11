import gc
from time import strftime
from collections import namedtuple
from src.algorithms.BaseAlgorithm import fromDirectory
from src.algorithms.HOG import HOG
from src.algorithms.FT import FT
from src.algorithms.SIFT import SIFT
from src.algorithms.SURF import SURF
from src.algorithms.BRIEF import BRIEF
from src.algorithms.ORB import ORB
from src.algorithms.FREAK import FREAK

size = "1280x720"

dataDir = "../data"
partsDir = f"{dataDir}/parts/{size}"
originalDir = f"{dataDir}/original/{size}"
outputDir = f"{dataDir}/experimentResults/{size}"

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
    for i in range(10):
        print(f"({strftime('%H:%M:%S')}) - Iteration {i + 1}")
        obj = a.type(parts=fromDirectory(partsDir), images=fromDirectory(originalDir), iteration=i)
        obj.process()
        obj.writeResults(f"{a.output}/{i}", includePart=True)
        obj.printResults(f"{a.output}/{i}_result.txt")
        gc.collect()

print(f"({strftime('%H:%M:%S')}) Ended")