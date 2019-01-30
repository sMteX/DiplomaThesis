import gc
from src.scripts.algorithms.BaseAlgorithm import fromDirectory
from src.scripts.algorithms.HOG import HOG
from src.scripts.algorithms.FT import FT
from src.scripts.algorithms.SIFT import SIFT
from src.scripts.algorithms.SURF import SURF
from src.scripts.algorithms.BRIEF import BRIEF
from src.scripts.algorithms.ORB import ORB
from src.scripts.algorithms.FREAK import FREAK

imagesDir = "../../data/images"
partsDir = f"{imagesDir}/testing/parts"
originalDir = f"{imagesDir}/original/300x300"
outputDir = f"{imagesDir}/testing/output/300x300"

algorithms = [
    {
        "name": "HOG",
        "class": HOG,
        "output": f"{outputDir}/hog"
    },
    {
        "name": "FT",
        "class": FT,
        "output": f"{outputDir}/ft"
    },
    {
        "name": "SIFT",
        "class": SIFT,
        "output": f"{outputDir}/sift"
    },
    {
        "name": "SURF",
        "class": SURF,
        "output": f"{outputDir}/surf"
    },
    {
        "name": "BRIEF",
        "class": BRIEF,
        "output": f"{outputDir}/fast_brief"
    },
    {
        "name": "ORB",
        "class": ORB,
        "output": f"{outputDir}/orb"
    },
    {
        "name": "FREAK",
        "class": FREAK,
        "output": f"{outputDir}/fast_freak"
    }
]

print("Started")

for a in algorithms:
    print(f"Algorithm: {a['name']}")
    if a["name"] == "HOG":
        print("Get yourself a cup of coffee, this will take almost an hour")
    for i in range(10):
        print(f"- Iteration {i}")
        obj = a["class"](parts=fromDirectory(partsDir), images=fromDirectory(originalDir))
        obj.process()
        obj.writeResults(f"{a['output']}/{i}", includePart=True)
        obj.printResults(f"{a['output']}/{i}_result.txt")
        gc.collect()

print("Ended")
