import cv2 as cv
import numpy as np
import os
from src.playground.algorithms.BaseAlgorithm import BaseAlgorithm
from timeit import default_timer as timer

class BaseKeypointAlgorithm(BaseAlgorithm):
    topMatches: int
    drawMatches: bool

    bf: cv.BFMatcher      # brute force matcher, its parameter "normType" varies with algorithms (L2 for SIFT/SURF, HAMMING for ORB, BRIEF, FREAK)

    def __init__(self, partType, parts, imageType, images, outputDir, topMatches=20, drawMatches=True):
        super().__init__(partType, parts, imageType, images, outputDir)
        self.topMatches = topMatches
        self.drawMatches = drawMatches

    def processImages(self):
        for filePath in self.imagePaths:
            img = cv.imread(filePath, 0)
            keypoints, descriptors, time = self.calculateDescriptor(img)
            ok, error = self.checkValidDetectOutput(keypoints, descriptors)
            if not ok:
                print(f"ERROR computing keypoints or descriptors for {filePath} ({error}), skipping...")
                continue
            self.diagnostics.times["imageDescriptor"].append(time)
            self.diagnostics.counts["imageDescriptorSize"].append(descriptors.size)
            self.imageData.append({
                "filePath": filePath,
                "keypoints": keypoints,
                "descriptors": descriptors
            })

    def processParts(self):
        for filePath in self.partPaths:
            partProcessTime = timer()
            img = cv.imread(filePath, 0)
            partSize = self.getSizeFromShape(img.shape)

            partKeypoints, partDescriptors, time = self.calculateDescriptor(img)
            ok = self.checkValidDetectOutput(partKeypoints, partDescriptors)
            if not ok[0]:
                print(f"ERROR computing keypoints or descriptors for {filePath} ({ok[1]}), skipping...")
                continue

            self.diagnostics.times["partDescriptor"].append(time)
            self.diagnostics.counts["partDescriptorSize"].append(partDescriptors.size)

            best = {
                "image": None,
                "distance": float("inf"),
                "topMatches": [],
            }

            allImageProcessTime = timer()
            for image in self.imageData:
                imageProcessTime = timer()
                matches = self.bf.match(partDescriptors, image["descriptors"])

                matches = sorted(matches, key=lambda x: x.distance)
                matches = matches[:self.topMatches]
                totalDistance = np.sum(np.asarray(list(map(lambda match: match.distance, matches))))

                if totalDistance < best["distance"]:
                    best["image"] = image
                    best["distance"] = totalDistance
                    best["topMatches"] = matches

                self.diagnostics.times["individualImageMatching"].append(timer() - imageProcessTime)

            end = timer()
            self.diagnostics.times["allImagesMatching"].append(end - allImageProcessTime)
            self.diagnostics.times["partProcess"].append(end - partProcessTime)

            # processing result
            bestMatch = best["topMatches"][0]
            bestKeypointPart = partKeypoints[bestMatch.queryIdx]
            bestKeypointImage = best["image"]["keypoints"][bestMatch.trainIdx]
            startX = np.round(bestKeypointImage.pt[0] - bestKeypointPart.pt[0]).astype(int)
            startY = np.round(bestKeypointImage.pt[1] - bestKeypointPart.pt[1]).astype(int)
            endX, endY = startX + partSize[0], startY + partSize[1]

            self.results.append({
                "outputFilePath": os.path.abspath(f"{self.outputDir}/{'kp_' if self.drawMatches else ''}{os.path.basename(filePath)}"),
                "imageFilePath": best["image"]["filePath"],
                "sX": startX,
                "sY": startY,
                "eX": endX,
                "eY": endY,
                # for DRAW_MATCHES=True
                "partFilePath": filePath,
                "partKeypoints": partKeypoints,
                "imageKeypoints": best["image"]["keypoints"],
                "topMatches": best["topMatches"]
            })

    # implement in child algorithms

    def calculateDescriptor(self, img) -> object:
        pass

    def writeResults(self):
        for result in self.results:
            resultImage = cv.imread(result["imageFilePath"])
            resultImage = cv.rectangle(img=resultImage,
                                       pt1=(result["sX"], result["sY"]),
                                       pt2=(result["eX"], result["eY"]),
                                       color=(0, 0, 255))
            if self.drawMatches:
                resultImage = cv.drawMatches(img1=cv.imread(result["partFilePath"]),
                                             keypoints1=result["partKeypoints"],
                                             img2=resultImage,
                                             keypoints2=result["imageKeypoints"],
                                             matches1to2=result["topMatches"],
                                             outImg=resultImage,
                                             flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            cv.imwrite(result["outputFilePath"], resultImage)

    def printResults(self):
        average = {
            "partDescriptor": self.avg(self.diagnostics.times["partDescriptor"]),
            "imageDescriptor": self.avg(self.diagnostics.times["imageDescriptor"]),
            "individualImageMatching": self.avg(self.diagnostics.times["individualImageMatching"]),
            "allImagesMatching": self.avg(self.diagnostics.times["allImagesMatching"]),
            "partProcess": self.avg(self.diagnostics.times["partProcess"]),

            "partDescriptorSize": self.avg(self.diagnostics.counts["partDescriptorSize"], self.AverageType.COUNT),
            "imageDescriptorSize": self.avg(self.diagnostics.counts["imageDescriptorSize"], self.AverageType.COUNT),
        }

        print(f"Total time [ms]: {self.diagnostics.totalTime}")
        print("Average times [ms]:")
        print(f"    - Keypoint and descriptor computing for a part: {average['partDescriptor']}")
        print(f"    - Keypoint and descriptor computing for an image: {average['imageDescriptor']}")
        print(f"    - Matching part with individual image: {average['individualImageMatching']}")
        print(f"    - Matching part with all images: {average['allImagesMatching']}")
        print(f"    - Processing entire part: {average['partProcess']}\n")
        print(f"Average part descriptor size: {average['partDescriptorSize']}")
        print(f"Average image descriptor size: {average['imageDescriptorSize']}")

    @staticmethod
    def checkValidDetectOutput(keypoints, descriptors):
        if len(keypoints) == 0:
            return False, "No keypoints detected"
        elif descriptors is None or descriptors.size == 0:
            return False, "No descriptors computed"
        return True, ""