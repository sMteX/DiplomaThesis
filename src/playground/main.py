def getSubsetsFromImage(partSize, imageSize):
    pW, pH = partSize
    iW, iH = imageSize
    y = 0
    while y + pH < iH:
        x = 0
        while x + pW < iW:
            yield x, y
            x = x + 1
        y = y + 1

image = (20, 10)
part = (5, 3)
for point in getSubsetsFromImage(part, image):
    print(f"{point} to {(point[0] + part[0], point[1] + part[1])}")