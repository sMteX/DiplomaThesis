import cv2 as cv
import numpy as np
import os
from src.playground.algorithms.BaseAlgorithm import BaseAlgorithm
from timeit import default_timer as timer

class BaseHogFT(BaseAlgorithm):
    def __init__(self, partType, parts, imageType, images, outputDir):
        super().__init__(partType, parts, imageType, images, outputDir)
        self.diagnostics.counts["subsets"] = []

    def processImages(self):
        for filename in self.imagePaths:
            img = cv.imread(filename, 0)
            descriptor, time = self.calculateDescriptor(img)
            self.diagnostics.times["imageDescriptor"].append(time)
            self.diagnostics.counts["imageDescriptorSize"].append(descriptor.size)
            self.imageData.append({
                "filename": filename,
                "descriptor": descriptor
            })

    def processParts(self):
        for i, filename in enumerate(self.partPaths):
            partProcessTime = timer()
            img = cv.imread(filename, 0)
            partSize = self.getSizeFromShape(img.shape)

            partDescriptor, time = self.calculateDescriptor(img)
            self.diagnostics.times["partDescriptor"].append(time)
            self.diagnostics.counts["partDescriptorSize"].append(partDescriptor.size)

            best = {
                "distance": float("inf"),  # default to +infinity
                "filename": "",
                "sX": -1,
                "sY": -1,
                "eX": -1,
                "eY": -1
            }

            for image in self.imageData:
                imageProcessTime = timer()
                subsets = 0
                windowSize, imageSize, stepSize = self.getSubsetParams(partDescriptor, image["descriptor"])
                for startX, startY, endX, endY in self.getSubsets(windowSize, imageSize, stepSize):
                    subsets = subsets + 1
                    subset = self.getSubset(image["descriptor"], startX, startY, endX, endY)

                    distance = np.linalg.norm(subset - partDescriptor)  # should calculate the euclidean distance

                    if distance < best["distance"]:
                        scaleX, scaleY = self.getResultPointScale()
                        best["distance"] = distance
                        best["filename"] = image["filename"]
                        best["sX"] = startX * scaleX
                        best["sY"] = startY * scaleY
                        best["eX"] = best["sX"] + partSize[0]
                        best["eY"] = best["sY"] + partSize[1]

                self.diagnostics.times["imageProcess"].append(timer() - imageProcessTime)
                self.diagnostics.counts["subsets"].append(subsets)

            self.diagnostics.times["partProcess"].append(timer() - partProcessTime)

            self.results.append({
                "outputFilename": os.path.abspath(f"{self.outputDir}/{i}.jpg"),
                "imageFilename": best["filename"],
                "sX": best["sX"],
                "sY": best["sY"],
                "eX": best["eX"],
                "eY": best["eY"]
            })

    def calculateDescriptor(self, img) -> object:
        pass

    def getSubsetParams(self, partDescriptor, imageDescriptor) -> object:
        pass

    def getSubset(self, descriptor, startX, startY, endX, endY) -> np.ndarray:
        pass

    def getResultPointScale(self) -> object:
        pass

    def writeResults(self):
        for result in self.results:
            resultImage = cv.imread(result["imageFilename"])
            resultImage = cv.rectangle(resultImage,
                                       pt1=(result["sX"], result["sY"]),
                                       pt2=(result["eX"], result["eY"]),
                                       color=(0, 0, 255))
            cv.imwrite(result["outputFilename"], resultImage)

    def printResults(self):
        average = {
            "partDescriptor": self.avg(self.diagnostics.times["partDescriptor"]),
            "imageDescriptor": self.avg(self.diagnostics.times["imageDescriptor"]),
            "imageProcess": self.avg(self.diagnostics.times["imageProcess"]),
            "partProcess": self.avg(self.diagnostics.times["partProcess"]),

            "partDescriptorSize": self.avg(self.diagnostics.counts["partDescriptorSize"], self.AverageType.COUNT),
            "imageDescriptorSize": self.avg(self.diagnostics.counts["imageDescriptorSize"], self.AverageType.COUNT),
            "subsets": self.avg(self.diagnostics.counts["subsets"], self.AverageType.COUNT),
        }

        print(f"Total time [ms]: {self.diagnostics.totalTime}")
        print("Average times [ms]:")
        print(f"    - Descriptor computing for a part: {average['partDescriptor']}")
        print(f"    - Descriptor computing for a image: {average['imageDescriptor']}")
        print(f"    - Processing entire part: {average['partProcess']}\n")
        print(f"Average part descriptor size: {average['partDescriptorSize']}")
        print(f"Average image descriptor size: {average['imageDescriptorSize']}")