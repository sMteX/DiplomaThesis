# gather data from .txt files scattered in data/experimentResults/{size}
# put into xlsx in particular format just to be copied into the main data worksheet

import xlsxwriter
import os
import re

rootDir = "../../data/experimentResults/1280x720"

folderNames = ["hog", "ft", "sift", "surf", "fast_brief", "orb", "fast_freak"]
textData = { }

# first, read text files, store it into dictionary
for folder in folderNames:
    textData[folder] = {}
    for i in range(10):
        path = os.path.abspath(f"{rootDir}/{folder}/{i}_result.txt")
        textData[folder][i] = [line.rstrip('\n') for line in open(path)]

print("Reading files done")

parsedData = {}
p = re.compile("(.+): (\d+\.\d+)")

# second, parse the lines from the dictionary, extracting only the numbers from the lines (and ignoring part with subsets and descriptor sizes)
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

print("Parsing done")

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
    """
    Returns the target cell for given algorithm, iteration and data index (measured variable)
    """
    return f"{algorithmColumnMap[algorithm]}{iteration * 9 + dataIndex + 1}"

excelData = []

# third, process the parsed data, and store pairs of (cell, data)
for a, ad in parsedData.items():
    for iteration, numbers in ad.items():
        for dataIndex, data in enumerate(numbers):
            excelData.append((cell(a, iteration, dataIndex), data.replace(".", ",")))

print("Processed data")

# create Excel sheet, write all stored data, save it
wb = xlsxwriter.Workbook('gatheredData1280x720.xlsx')
s = wb.add_worksheet()

for cellName, value in excelData:
    s.write(cellName, value)

wb.close()