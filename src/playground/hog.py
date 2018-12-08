import cv2 as cv
import numpy as np
import os

height = 160
width = 60

# parametry pro HOGDescriptor
cellSide = 12
cellSize = (cellSide, cellSide)     # h x w
blockSize = (cellSide * 2, cellSide * 2)    # h x w
blockStride = cellSize
winSize = (width // cellSize[1] * cellSize[1], height // cellSize[0] * cellSize[0])
nbins = 9
# defaultne nastavene parametry
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True

positiveTrainDir = "trainBig/positive"
negativeTrainDir = "trainBig/negative"
positiveTestDir = "testBig/positive"
negativeTestDir = "testBig/negative"

positiveTrainCount = len(os.listdir(positiveTrainDir))
negativeTrainCount = len(os.listdir(negativeTrainDir))
positiveTestCount = len(os.listdir(positiveTestDir))
negativeTestCount = len(os.listdir(negativeTestDir))

maxSVMiterations = 1000
epsilon = 1e-6

realWidth, realHeight = winSize
positives = np.empty([positiveTrainCount, realHeight, realWidth])
label_positive = np.ones([positiveTrainCount])
negatives = np.empty([negativeTrainCount, realHeight, realWidth])
label_negative = np.zeros([negativeTrainCount])
label_negative[:] = -1

for i, filename in enumerate(os.listdir(positiveTrainDir)):
    img = cv.imread(f"{positiveTrainDir}/{filename}", 0)
    # resize na stejnou velikost jako je winSize pro HOG
    img = cv.resize(img, winSize)
    positives[i] = img

for i, filename in enumerate(os.listdir(negativeTrainDir)):
    img = cv.imread(f"{negativeTrainDir}/{filename}", 0)
    img = cv.resize(img, winSize)
    negatives[i] = img

trainDataImg = np.concatenate((positives, negatives))
labels = np.concatenate((label_positive, label_negative))

hog = cv.HOGDescriptor(winSize,
                        blockSize,
                        blockStride,
                        cellSize,
                        nbins,
                        derivAperture,
                        winSigma,
                        histogramNormType,
                        L2HysThreshold,
                        gammaCorrection,
                        nlevels,
                        signedGradients)
trainData = np.empty([positiveTrainCount + negativeTrainCount, hog.getDescriptorSize(), 1])
trainDataImg = np.uint8(trainDataImg)
for i, img in enumerate(trainDataImg):
    trainData[i] = hog.compute(img)

trainData = np.float32(trainData.reshape(positiveTrainCount + negativeTrainCount, hog.getDescriptorSize()))
labels = np.int32(labels)

svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
# default kernel je RBF
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, maxSVMiterations, epsilon))
svm.trainAuto(trainData, cv.ml.ROW_SAMPLE, labels)

test_positive = np.empty([positiveTestCount, 1, hog.getDescriptorSize()])
for i, filename in enumerate(os.listdir(positiveTestDir)):
    img = cv.imread(f"{positiveTestDir}/{filename}", 0)
    img = cv.resize(img, winSize)
    components = hog.compute(img)
    components = components.reshape(1, hog.getDescriptorSize())
    test_positive[i] = components

test_negative = np.empty([negativeTestCount, 1, hog.getDescriptorSize()])
for i, filename in enumerate(os.listdir(negativeTestDir)):
    img = cv.imread(f"{negativeTestDir}/{filename}", 0)
    img = cv.resize(img, winSize)
    components = hog.compute(img)
    components = components.reshape(1, hog.getDescriptorSize())
    test_negative[i] = components

# print("Positive classification times", positiveClassificationTimes, "Negative classification times", negativeClassificationTimes)
test_positive = np.float32(test_positive)
test_negative = np.float32(test_negative)

correctPositive = 0
for i, sample in enumerate(test_positive):
    result = svm.predict(sample)[1]
    # index v test_positive by mel presne souhlasit s indexem souboru v tom adresari
    img = cv.imread(f"{positiveTestDir}/{os.listdir(positiveTestDir)[i]}")
    if result == 1:
        correctPositive += 1
        cv.imwrite(f"resultsHOG/correct/positive_{i}.jpg", img)
    else:
        cv.imwrite(f"resultsHOG/incorrect/should_be_positive_{i}.jpg", img)

correctNegative = 0
for i, sample in enumerate(test_negative):
    img = cv.imread(f"{negativeTestDir}/{os.listdir(negativeTestDir)[i]}")
    result = svm.predict(sample)[1]
    if result == -1:
        correctNegative += 1
        cv.imwrite(f"resultsHOG/correct/negative_{i}.jpg", img)
    else:
        cv.imwrite(f"resultsHOG/incorrect/should_be_negative_{i}.jpg", img)


totalTestCount = positiveTestCount + negativeTestCount
totalCorrect = correctPositive + correctNegative
accuracy = totalCorrect / totalTestCount
print("Results\n----------------------------------")
print(f"HOG cell side: {cellSide}")
print(f"HOG descriptor size: {hog.getDescriptorSize()}")
print(f"Resulting accuracy: {accuracy * 100} % ({totalCorrect}/{totalTestCount} samples)")
print(f"{correctPositive}/{positiveTestCount} positive examples")
print(f"{correctNegative}/{negativeTestCount} negative examples")
