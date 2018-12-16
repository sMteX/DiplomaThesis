import cv2 as cv
import numpy as np
import os
from src.playground.algorithms.BaseAlgorithm import BaseAlgorithm
from timeit import default_timer as timer

class Hog(BaseAlgorithm):
    class Diagnostics:
        times = {
            "partDescriptor": [],
            "imageDescriptor": [],
            "distanceComputing": [],
            "imageProcess": [],
            "partProcess": [],
        }
        counts = {
            "partDescriptorSize": [],
            "imageDescriptorSize": [],
            "subsets": [],
        }

    # parameters for HOGDescriptor
    cellSide = 4
    cellSize = (cellSide, cellSide)  # w x h
    blockSize = (cellSide * 2, cellSide * 2)  # w x h
    blockStride = cellSize
    nBins = 9
    # other parameters
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nLevels = 64
    signedGradients = True

    imageData = []
    results = []
    diagnostics = Diagnostics()

    def __init__(self, partType, parts, imageType, images, outputDir, cellSide=None):
        super().__init__(partType, parts, imageType, images, outputDir)
        if cellSide is not None:
            self.cellSide = cellSide
            self.cellSize = (self.cellSide, self.cellSide)  # w x h
            self.blockSize = (self.cellSide * 2, self.cellSide * 2)  # w x h
            self.blockStride = self.cellSize

    def processImages(self):
        print("Hog.processImages() called")
        for filename in self.imagePaths:
            img = cv.imread(filename, 0)
            croppedSize = self.getCroppedSize(self.cellSize, self.getSizeFromShape(img.shape))
            img = cv.resize(img, croppedSize)
            hog = cv.HOGDescriptor(
                croppedSize,
                self.blockSize,
                self.blockStride,
                self.cellSize,
                self.nBins,
                self.derivAperture,
                self.winSigma,
                self.histogramNormType,
                self.L2HysThreshold,
                self.gammaCorrection,
                self.nLevels,
                self.signedGradients
            )

            t = timer()
            descriptor = hog.compute(img)
            self.diagnostics.times["imageDescriptor"].append(timer() - t)

            newSize = self.getReshapedSize(hog.getDescriptorSize(), croppedSize, self.blockSize, self.blockStride)
            reshapedDescriptor = np.reshape(descriptor, newSize)

            self.diagnostics.counts["imageDescriptorSize"].append(reshapedDescriptor.size)

            self.imageData.append({
                "filename": filename,
                "descriptors": reshapedDescriptor
            })

    def processParts(self):
        print("Hog.processParts() called")
        for i, filename in enumerate(self.partPaths):
            partProcessTime = timer()
            img = cv.imread(filename, 0)
            partSize = self.getSizeFromShape(img.shape)
            # resize it to multiple of cellSize
            croppedSize = self.getCroppedSize(self.cellSize, partSize)
            img = cv.resize(img, croppedSize)
            # construct a HOG descriptor for given size
            hog = cv.HOGDescriptor(
                croppedSize,
                self.blockSize,
                self.blockStride,
                self.cellSize,
                self.nBins,
                self.derivAperture,
                self.winSigma,
                self.histogramNormType,
                self.L2HysThreshold,
                self.gammaCorrection,
                self.nLevels,
                self.signedGradients
            )

            t = timer()
            originalPartDescriptor = hog.compute(img)
            self.diagnostics.times["partDescriptor"].append(timer() - t)

            newPartSize = self.getReshapedSize(hog.getDescriptorSize(), croppedSize, self.blockSize, self.blockStride)
            partDescriptor = np.reshape(originalPartDescriptor, newPartSize)

            self.diagnostics.counts["partDescriptorSize"].append(partDescriptor.size)

            best = {
                "distance": float("inf"),  # default to +infinity
                "filename": "",
                "sX": -1,
                "sY": -1,
                "eX": -1,
                "eY": -1
            }

            # for each part, iterate through original files and all parts of the images with the same size
            for image in self.imageData:
                imageProcessTime = timer()
                subsets = 0
                for startX, startY, endX, endY in self.getSubsets(partDescriptor.shape[:2],
                                                                  image["descriptors"].shape[:2],
                                                                  (1, 1)):
                    subsets = subsets + 1
                    subset = image["descriptors"][startX:endX, startY:endY, :, :]

                    t = timer()
                    distance = np.linalg.norm(subset - partDescriptor)  # should calculate the euclidean distance
                    self.diagnostics.times["distanceComputing"].append(timer() - t)

                    if distance < best["distance"]:
                        best["distance"] = distance
                        best["filename"] = image["filename"]
                        best["sX"] = startX * self.blockStride[0]
                        best["sY"] = startY * self.blockStride[1]
                        best["eX"] = best["sX"] + partSize[0]
                        best["eY"] = best["sY"] + partSize[1]
                self.diagnostics.times["imageProcess"].append(timer() - imageProcessTime)
                self.diagnostics.counts["subsets"].append(subsets)

            self.diagnostics.times["partProcess"].append(timer() - partProcessTime)

            self.results.append({
                "outputFilename": os.path.abspath(f"{self.outputDir}/{i}.jpg"), # os.path.basename() ?
                "imageFilename": best["filename"],
                "sX": best["sX"],
                "sY": best["sY"],
                "eX": best["eX"],
                "eY": best["eY"]
            })

    def writeResults(self):
        print("Hog.writeResults() called")
        for result in self.results:
            resultImage = cv.imread(result["imageFilename"])
            resultImage = cv.rectangle(resultImage,
                                       pt1=(result["sX"], result["sY"]),
                                       pt2=(result["eX"], result["eY"]),
                                       color=(0, 0, 255))
            cv.imwrite(result["outputFilename"], resultImage)
            
    def printResults(self):
        print("Hog.printResults() called")
        average = {
            "partDescriptor": self.avg(self.diagnostics.times["partDescriptor"]),
            "imageDescriptor": self.avg(self.diagnostics.times["imageDescriptor"]),
            "distanceComputing": self.avg(self.diagnostics.times["distanceComputing"]),
            "imageProcess": self.avg(self.diagnostics.times["imageProcess"]),
            "partProcess": self.avg(self.diagnostics.times["partProcess"]),

            "partDescriptorSize": self.avg(self.diagnostics.counts["partDescriptorSize"], self.AverageType.COUNT),
            "imageDescriptorSize": self.avg(self.diagnostics.counts["imageDescriptorSize"], self.AverageType.COUNT),
            "subsets": self.avg(self.diagnostics.counts["subsets"], self.AverageType.COUNT),
        }

        print(f"Total time [ms]: {self.totalTime}")
        print("Average times [ms]:")
        print(f"    - Descriptor computing for a part: {average['partDescriptor']}")
        print(f"    - Descriptor computing for a image: {average['imageDescriptor']}")
        print(f"    - Calculating distance between descriptors: {average['distanceComputing']}")
        print(f"    - Processing all subsets (average {average['subsets']} subsets) for a single image: {average['imageProcess']}")
        print(f"    - Processing entire part: {average['partProcess']}\n")
        print(f"Average part descriptor size: {average['partDescriptorSize']}")
        print(f"Average image descriptor size: {average['imageDescriptorSize']}")

    @staticmethod
    def getReshapedSize(descriptorSize, winSize, blockSize, blockStride):
        w = (winSize[0] - blockSize[0]) / blockStride[0] + 1
        h = (winSize[1] - blockSize[1]) / blockStride[1] + 1
        dSize = descriptorSize / w / h
        return int(w), int(h), int(dSize), 1