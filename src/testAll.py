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

imagesDir = "../../data/images"
partsDir = f"{imagesDir}/testing/parts/{size}"
originalDir = f"{imagesDir}/original/{size}"
outputDir = f"{imagesDir}/testing/output/{size}"

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
    # if a.name == "HOG" and size == "1280x720":
        # with the setting that's in place for HOG (and changing it to make HOG better wouldn't be fair?)
        # it's doing ENORMOUS amount of descriptor comparisons per part and image for 1280x720
        # I've run some tests to figure out why it can't finish a single iteration in 4+ hours, and turns out
        # it's doing MILLIONS of comparisons for each part (actually 2 593 221 on average)..
        # and even though calculating the np.linalg.norm() takes around 0.2ms on average,
        # that still means an average of around 8.64 minutes per part => 432 minutes per iteration (7 hours 12 minutes)
        # and I'd need 10 of those - NOPE

        # UPDATE: it seems like it doesn't matter and I have to do it :/

        # continue
    for i in range(10):
        print(f"({strftime('%H:%M:%S')}) - Iteration {i + 1}")
        obj = a.type(parts=fromDirectory(partsDir), images=fromDirectory(originalDir), iteration=i)
        obj.process()
        obj.writeResults(f"{a.output}/{i}", includePart=True)
        obj.printResults(f"{a.output}/{i}_result.txt")
        gc.collect()

print(f"({strftime('%H:%M:%S')}) Ended")