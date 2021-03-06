import cv2 as cv
import numpy as np
import os
from time import strftime
from src.algorithms.BaseAlgorithm import BaseAlgorithm
from timeit import default_timer as timer

STEPS_PER_ITERATION = 50 * 75
TOTAL_STEPS = STEPS_PER_ITERATION * 10

class BaseHogFT(BaseAlgorithm):
    def __init__(self, parts, images, iteration = None):
        super().__init__(parts, images, iteration)

    def processImages(self):
        print("---------")
        for i, image in enumerate(self.images):
            print(f"(Iteration {self.iteration + 1}, {strftime('%H:%M:%S')}) Preprocessing image {i + 1}")
            # convert image to gray, calculate descriptor for it (somehow)
            img = cv.cvtColor(image.colorImage, cv.COLOR_BGR2GRAY)
            descriptor, time = self.calculateDescriptor(img)

            self.diagnostics.times.imageDescriptor.append(time)
            self.diagnostics.counts.imageDescriptorSize.append(descriptor.size)
            # save the descriptor and image it belongs to
            self.imageData.append({
                "colorImage": image.colorImage,
                "path": image.filePath,
                "descriptor": descriptor
            })

    def processParts(self):
        for i, part in enumerate(self.parts):
            print("---------")
            progress = self.iteration * STEPS_PER_ITERATION + i * 75
            print(f"(Iteration {self.iteration + 1}, {strftime('%H:%M:%S')}) Processing part {i + 1}/{len(self.parts)} ({(100 * (progress / TOTAL_STEPS)):.2f} %)")
            partProcessTime = timer()
            # convert part to gray, calculate descriptor for it
            img = cv.cvtColor(part.colorImage, cv.COLOR_BGR2GRAY)
            partSize = self.getSizeFromShape(img.shape)

            partDescriptor, time = self.calculateDescriptor(img)

            self.diagnostics.times.partDescriptor.append(time)
            self.diagnostics.counts.partDescriptorSize.append(partDescriptor.size)

            # structure for storing the best result
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
            for j, image in enumerate(self.imageData):
                progress = self.iteration * STEPS_PER_ITERATION + i * 75 + j
                print(f"- (Iteration {self.iteration + 1}, {strftime('%H:%M:%S')}) Pairing part {i + 1} with image {j + 1}/{len(self.imageData)} ({(100 * (progress / TOTAL_STEPS)):.2f} %)")
                imageProcessTime = timer()
                subsets = 0
                # extract info about how subsets from the image should be gotten
                windowSize, imageSize, stepSize = self.getSubsetParams(partDescriptor, image["descriptor"])
                # iterate through all subsets from the image with the same size as the searched part
                for startX, startY, endX, endY in self.getSubsets(windowSize, imageSize, stepSize):
                    subsets = subsets + 1
                    subset = self.getSubset(image["descriptor"], startX, startY, endX, endY)

                    distance = np.linalg.norm(subset - partDescriptor)  # should calculate the euclidean distance

                    if distance < best["distance"]:
                        # get "scale" which translates (x,y) pair from the descriptor point of view to pixel (x,y)
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
        """
        Calculates descriptor for given image

        :return: Tuple (descriptor, time) - time it took to calculate the descriptor
        """
        pass

    def getSubsetParams(self, partDescriptor, imageDescriptor) -> object:
        """
        Extract info about how subsets from the image should be gotten

        :return: Tuple (windowSize, imageSize, stepSize), where all sizes are tuples (int, int)
        """
        pass

    def getSubset(self, descriptor, startX, startY, endX, endY) -> np.ndarray:
        """
        Gets a subset of target descriptor at given start and end positions, returns a Numpy array
        """
        pass

    def getResultPointScale(self) -> object:
        """
        Returns a "scale" which translates (x,y) pair from the descriptor point of view to pixel (x,y)
        :return: Tuple (scaleX, scaleY) - (int, int)
        """
        pass

    def writeSingleResult(self, result, path, includePart=False):
        path = os.path.abspath(path)
        resultImage = result.image.copy()
        resultImage = cv.rectangle(resultImage,
                                   pt1=result.start,
                                   pt2=result.end,
                                   color=(0, 0, 255))

        if includePart:
            # create a new image with width = part width + image width, height = image height
            out = np.zeros((resultImage.shape[0], resultImage.shape[1] + result.part.shape[1], 3), np.uint8)
            # copy part to top left corner (area under it will be black)
            out[0:result.part.shape[0], 0:result.part.shape[1]] = result.part
            # copy result image next to it
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