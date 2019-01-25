import cv2 as cv
import numpy as np
import os
from src.scripts.algorithms.BaseAlgorithm import BaseAlgorithm
from timeit import default_timer as timer

class BaseHogFT(BaseAlgorithm):
    def __init__(self, parts, images):
        super().__init__(parts, images)

    def processImages(self):
        for image in self.images:
            img = cv.cvtColor(image.colorImage, cv.COLOR_BGR2GRAY)
            descriptor, time = self.calculateDescriptor(img)
            self.diagnostics.times.imageDescriptor.append(time)
            self.diagnostics.counts.imageDescriptorSize.append(descriptor.size)
            self.imageData.append({
                "colorImage": image.colorImage,
                "descriptor": descriptor
            })

    def processParts(self):
        for part in self.parts:
            partProcessTime = timer()
            img = cv.cvtColor(part.colorImage, cv.COLOR_BGR2GRAY)
            partSize = self.getSizeFromShape(img.shape)

            partDescriptor, time = self.calculateDescriptor(img)
            self.diagnostics.times.partDescriptor.append(time)
            self.diagnostics.counts.partDescriptorSize.append(partDescriptor.size)

            best = {
                "distance": float("inf"),  # default to +infinity
                "colorImage": None,
                "sX": -1,
                "sY": -1,
                "eX": -1,
                "eY": -1
            }

            allImageProcessTime = timer()
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
                        best["colorImage"] = image["colorImage"]
                        best["sX"] = startX * scaleX
                        best["sY"] = startY * scaleY
                        best["eX"] = best["sX"] + partSize[0]
                        best["eY"] = best["sY"] + partSize[1]

                self.diagnostics.times.individualImageMatching.append(timer() - imageProcessTime)
                self.diagnostics.counts.subsets.append(subsets)

            end = timer()
            self.diagnostics.times.allImagesMatching.append(end - allImageProcessTime)
            self.diagnostics.times.partProcess.append(end - partProcessTime)

            self.results.append(self.MatchingResult(part=part.colorImage,
                                                    image=best["colorImage"],
                                                    start=(best["sX"], best["sY"]),
                                                    end=(best["eX"], best["eY"])))

    # implement in child algorithms

    def calculateDescriptor(self, img) -> object:
        pass

    def getSubsetParams(self, partDescriptor, imageDescriptor) -> object:
        pass

    def getSubset(self, descriptor, startX, startY, endX, endY) -> np.ndarray:
        pass

    def getResultPointScale(self) -> object:
        pass

    def writeResults(self, directory):
        path = os.path.abspath(directory)
        for i, result in enumerate(self.results):
            resultImage = result.image.copy()
            resultImage = cv.rectangle(resultImage,
                                       pt1=result.start,
                                       pt2=result.end,
                                       color=(0, 0, 255))
            cv.imwrite(f"{path}/{i}.jpg", resultImage)

    def printResults(self):
        average = {
            "partDescriptor": self.avg(self.diagnostics.times.partDescriptor),
            "imageDescriptor": self.avg(self.diagnostics.times.imageDescriptor),
            "individualImageMatching": self.avg(self.diagnostics.times.individualImageMatching),
            "allImagesMatching": self.avg(self.diagnostics.times.allImagesMatching),
            "partProcess": self.avg(self.diagnostics.times.partProcess),

            "partDescriptorSize": self.avg(self.diagnostics.counts.partDescriptorSize, self.AverageType.COUNT),
            "imageDescriptorSize": self.avg(self.diagnostics.counts.imageDescriptorSize, self.AverageType.COUNT),
            "subsets": self.avg(self.diagnostics.counts.subsets, self.AverageType.COUNT),
        }

        print(f"Total time [ms]: {self.diagnostics.totalTime}")
        print("Average times [ms]:")
        print(f"    - Descriptor computing for a part: {average['partDescriptor']}")
        print(f"    - Descriptor computing for a image: {average['imageDescriptor']}")
        print(f"    - Matching part with individual image: {average['individualImageMatching']}")
        print(f"    - Matching part with all images: {average['allImagesMatching']}")
        print(f"    - Processing entire part: {average['partProcess']}\n")
        print(f"Average part descriptor size: {average['partDescriptorSize']}")
        print(f"Average image descriptor size: {average['imageDescriptorSize']}")
        print(f"Average subsets in image: {average['subsets']}")