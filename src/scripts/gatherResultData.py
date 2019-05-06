# gather data from .txt files scattered in data/images/testing/output
# put into xlsx in particular format just to be copied into the main data worksheet

import xlsxwriter
import os
import re

rootDir = "../../data/images/testing/output/1280x720"

folderNames = ["hog", "ft", "sift", "surf", "fast_brief", "orb", "fast_freak"]
textData = { }

for folder in folderNames:
    textData[folder] = {}
    for i in range(10):
        path = os.path.abspath(f"{rootDir}/{folder}/{i}_result.txt")
        textData[folder][i] = [line.rstrip('\n') for line in open(path)]

print("reading done")

parsedData = {}
p = re.compile("(.+): (\d+\.\d+)")

for algorithm, algorithmDict in textData.items():
    parsedData[algorithm] = {}
    for i, lines in algorithmDict.items():
        parsedData[algorithm][i] = []
        for line in lines:
            m = p.match(line)
            if not m is None:
                if "subsets" not in m.group(1):
                    parsedData[algorithm][i].append(m.group(2))
        parsedData[algorithm][i] = parsedData[algorithm][i][:-2] # remove last 2 values (descriptor sizes)

print("parsing done")

algorithmColumnMap = {
    "hog": "A",
    "ft": "B",
    "sift": "C",
    "surf": "D",
    "fast_brief": "E",
    "orb": "F",
    "fast_freak": "G"
}

def cell(algorithm, iteration, dataIndex):
    return f"{algorithmColumnMap[algorithm]}{iteration * 9 + dataIndex + 1}"

excelData = []

for a, ad in parsedData.items():
    for iteration, numbers in ad.items():
        for dataIndex, data in enumerate(numbers):
            excelData.append((cell(a, iteration, dataIndex), data.replace(".", ",")))

print("processed")

wb = xlsxwriter.Workbook('gatheredData1280x720.xlsx')
s = wb.add_worksheet()

for cellName, value in excelData:
    s.write(cellName, value)

wb.close()