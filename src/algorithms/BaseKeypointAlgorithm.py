import cv2 as cv
import numpy as np
import os
from src.algorithms.BaseAlgorithm import BaseAlgorithm
from timeit import default_timer as timer

class BaseKeypointAlgorithm(BaseAlgorithm):
    bf: cv.BFMatcher      # brute force matcher, its parameter "normType" varies with algorithms (L2 for SIFT/SURF, HAMMING for ORB, BRIEF, FREAK)

    def __init__(self, parts, images, topMatches=20, drawMatches=True, iteration=None):
        super().__init__(parts, images, iteration)
        # how many keypoint matches should be taken into account when looking for the best result (also how many matches should be drawn in the result)
        self.topMatches = topMatches
        # if matches should be visualized in the result or not
        self.drawMatches = drawMatches

    def processImages(self):
        for image in self.images:
            # converts image to gray, calculates keypoints and descriptors for them (somehow)
            img = cv.cvtColor(image.colorImage, cv.COLOR_BGR2GRAY)
            keypoints, descriptors, time = self.calculateDescriptor(img)
            # sometimes, there are no keypoints found, check if the output is valid, otherwise skip this image
            ok, error = self.checkValidDetectOutput(keypoints, descriptors)
            if not ok:
                print(f"ERROR computing keypoints or descriptors for {image.filePath} ({error}), skipping...")
                continue
            self.diagnostics.times.imageDescriptor.append(time)
            self.diagnostics.counts.imageDescriptorSize.append(descriptors.size)
            # save keypoints, descriptors and image itself to memory
            self.imageData.append({
                "colorImage": image.colorImage,
                "path": image.filePath,
                "keypoints": keypoints,
                "descriptors": descriptors
            })

    def processParts(self):
        for part in self.parts:
            partProcessTime = timer()
            # convert part to gray, calculate keypoints and descriptors for it
            img = cv.cvtColor(part.colorImage, cv.COLOR_BGR2GRAY)
            partSize = self.getSizeFromShape(img.shape)

            partKeypoints, partDescriptors, time = self.calculateDescriptor(img)
            # check output for validity, skip part otherwise
            ok, error = self.checkValidDetectOutput(partKeypoints, partDescriptors)
            if not ok:
                print(f"ERROR computing keypoints or descriptors for {part.filePath} ({error}), skipping...")
                continue

            self.diagnostics.times.partDescriptor.append(time)
            self.diagnostics.counts.partDescriptorSize.append(partDescriptors.size)

            # structure for storing the best result
            # topMatches contains the keypoint matches
            # which themselves contain the source and target image + coordinates
            best = {
                "image": None,
                "distance": float("inf"),
                "topMatches": [],
            }

            allImageProcessTime = timer()
            for image in self.imageData:
                imageProcessTime = timer()
                # brute-force match all part descriptors with image descriptors
                matches = self.bf.match(partDescriptors, image["descriptors"])

                # sort matches by distance, take certain amount of best ones and sum the distances
                matches = sorted(matches, key=lambda x: x.distance)
                matches = matches[:self.topMatches]
                totalDistance = np.sum(np.asarray(list(map(lambda match: match.distance, matches))))

                if totalDistance < best["distance"]:
                    best["image"] = image
                    best["distance"] = totalDistance
                    best["topMatches"] = matches

                self.diagnostics.times.individualImageMatching.append(timer() - imageProcessTime)

            end = timer()
            self.diagnostics.times.allImagesMatching.append(end - allImageProcessTime)
            self.diagnostics.times.partProcess.append(end - partProcessTime)

            # process result - extract start and end points of the rectangle, where part was found
            bestMatch = best["topMatches"][0]
            bestKeypointPart = partKeypoints[bestMatch.queryIdx]
            bestKeypointImage = best["image"]["keypoints"][bestMatch.trainIdx]
            startX = np.round(bestKeypointImage.pt[0] - bestKeypointPart.pt[0]).astype(int)
            startY = np.round(bestKeypointImage.pt[1] - bestKeypointPart.pt[1]).astype(int)
            endX, endY = startX + partSize[0], startY + partSize[1]

            self.results.append(self.MatchingResult(part=part.colorImage,
                                                    image=best["image"]["colorImage"],
                                                    partPath=part.filePath,
                                                    imagePath=best["image"]["path"],
                                                    start=(startX, startY),
                                                    end=(endX, endY),
                                                    partKeypoints=partKeypoints,
                                                    imageKeypoints=best["image"]["keypoints"],
                                                    topMatches=best["topMatches"]))

    # implement in child algorithms

    def calculateDescriptor(self, img) -> object:
        """
        Calculates keypoints and descriptors for given image
        :return: Tuple (keypoints, descriptors, time)
        """
        pass

    def writeSingleResult(self, result, path, includePart=False):
        path = os.path.abspath(path)
        resultImage = result.image.copy()
        resultImage = cv.rectangle(img=resultImage,
                                   pt1=result.start,
                                   pt2=result.end,
                                   color=(0, 0, 255))
        if self.drawMatches:
            # OpenCV's drawMatches() does the same as the elif branch under this one
            # combines part and target image into one and draws matches between corresponding keypoints
            resultImage = cv.drawMatches(img1=result.part.copy(),
                                         keypoints1=result.partKeypoints,
                                         img2=resultImage,
                                         keypoints2=result.imageKeypoints,
                                         matches1to2=result.topMatches,
                                         outImg=resultImage,
                                         flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            cv.imwrite(path, resultImage)
        elif includePart:
            # create a new image which combines part and target image, save to file
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
        }

        lines = [
            f"Total time [ms]: {self.diagnostics.totalTime}\n",
            "Average times [ms]:\n",
            f"    - Keypoint and descriptor computing for a part: {average['partDescriptor']}\n",
            f"    - Keypoint and descriptor computing for an image: {average['imageDescriptor']}\n",
            f"    - Matching part with individual image: {average['individualImageMatching']}\n",
            f"    - Matching part with all images: {average['allImagesMatching']}\n",
            f"    - Processing entire part: {average['partProcess']}\n\n",
            f"Average part descriptor size: {average['partDescriptorSize']}\n",
            f"Average image descriptor size: {average['imageDescriptorSize']}"
        ]

        if not filename is None:
            with open(filename, "w") as file:
                for line in lines:
                    file.write(line)
        else:
            for line in lines:
                print(line)


    @staticmethod
    def checkValidDetectOutput(keypoints, descriptors):
        if len(keypoints) == 0:
            return False, "No keypoints detected"
        elif descriptors is None or descriptors.size == 0:
            return False, "No descriptors computed"
        return True, ""