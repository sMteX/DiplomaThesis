import cv2 as cv
import numpy as np
import os
from types import LambdaType
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
                "path": image.filePath,
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
                "path": None,
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
                        best["path"] = image["path"]
                        best["sX"] = startX * scaleX
                        best["sY"] = startY * scaleY
                        best["eX"] = best["sX"] + partSize[0]
                        best["eY"] = best["sY"] + partSize[1]

                self.diagnostics.times.individualImageMatching.append(timer() - imageProcessTime)
                self.diagnostics.counts.subsets.append(subsets)

            end = timer()
            self.diagnostics.times.allImagesMatching.append(end - allImageProcessTime)
            self.diagnostics.times.partProcess.append(end - partProcessTime)

            # noinspection PyTypeChecker
            self.results.append(self.MatchingResult(part=part.colorImage,
                                                    image=best["colorImage"],
                                                    partPath=part.filePath,
                                                    imagePath=best["path"],
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

    def writeResults(self, target, includePart=False):
        isLambda = isinstance(target, LambdaType)
        for i, result in enumerate(self.results):
            path = os.path.abspath(f"{target}/{i}.jpg") if not isLambda else os.path.abspath(target(i))
            self.writeSingleResult(result, path, includePart)

    def writeSingleResult(self, result, path, includePart=False):
        path = os.path.abspath(path)
        resultImage = result.image.copy()
        resultImage = cv.rectangle(resultImage,
                                   pt1=result.start,
                                   pt2=result.end,
                                   color=(0, 0, 255))

        if includePart:
            out = np.zeros((resultImage.shape[0], resultImage.shape[1] + result.part.shape[1], 3), np.uint8)
            out[0:result.part.shape[0], 0:result.part.shape[1]] = result.part
            out[0:, result.part.shape[1]:] = resultImage
            cv.imwrite(path, out)
        else:
            cv.imwrite(path, resultImage)

    def printResults(self, filename=None):
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

        lines = [
            f"Total time [ms]: {self.diagnostics.totalTime}\n",
            "Average times [ms]:\n",
            f"    - Descriptor computing for a part: {average['partDescriptor']}\n",
            f"    - Descriptor computing for a image: {average['imageDescriptor']}\n",
            f"    - Matching part with individual image: {average['individualImageMatching']}\n",
            f"    - Matching part with all images: {average['allImagesMatching']}\n",
            f"    - Processing entire part: {average['partProcess']}\n\n",
            f"Average part descriptor size: {average['partDescriptorSize']}\n",
            f"Average image descriptor size: {average['imageDescriptorSize']}\n",
            f"Average subsets in image: {average['subsets']}"
        ]

        if not filename is None:
            with open(filename, "w") as file:
                for line in lines:
                    file.write(line)
        else:
            for line in lines:
                print(line)