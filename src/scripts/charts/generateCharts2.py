import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import FuncFormatter
import src.scripts.charts.data as data

OUTPUT_DIR = "./output"

# taken from Wikipedia
def toHSV(h, s, l):
    _h = h
    _v = l + s * min(l, 1 - l)
    if _v == 0:
        return _h, 0, _v
    else:
        return _h, 2 - (2 * l) / _v, _v

def getLValues(count):
    _min = 0.2
    _max = 0.9
    step = (_max - _min) / count / 2
    return np.arange(_min + step, _max, 2 * step)

def getColors(colors, count):
    LValues = getLValues(count)
    # for each L, return colors with given L, return array of arrays of colors (corresponding to different datasets)
    result = []
    for L in LValues:
        temp = []
        for c in colors:
            h, s = c
            temp.append(hsv_to_rgb(toHSV(h, s, L)))
        result.append(temp)
    return result

COLORS = ["#cc00cc", "#e00000", "#e07000", "#efcf00", "#009e02", "#006ce0", "#7300e0"]
# "HSL" values, missing the L (that gets calculated based on data count)
COLORS_HS = [
    (0.833, 1),
    (0, 1),
    (0.083, 1),
    (0.1416, 1),
    (0.333, 1),
    (0.5861, 1),
    (0.75, 1)
]

PERCENTAGE_FORMATTER = lambda y, _: f"{(100 * y):.0f} %"
SECOND_FORMATTER = lambda y, _: math.floor(y / 1000.0)

PICTURE_SIZE = (10, 6)
DEFAULT_MARGINS = {
    "left": 1.5,
    "right": 1.5,
    "bottom": 0.6
}
TOP_MARGIN_NO_TITLE = 0.1
TOP_MARGIN_TITLE = 0.3

def transformMargins(left, right, top, bottom, pictureSize):
    return {
        "left": left / pictureSize[0],
        "right": 1 - right / pictureSize[0],
        "top": 1 - top / pictureSize[1],
        "bottom": bottom / pictureSize[1]
    }

def _checkSettings(settings, required):
    for requiredArg in required:
        if not requiredArg in settings:
            raise TypeError(f"Required argument {requiredArg} not found in settings")

# possible settings (kwargs):
# required:
#   yAxisLabel
# optional:
#   title,
#   yAxisFormatter
def plotSingleAxisSingleData(data, filename=None, show=False, **settings):
    _checkSettings(settings, ["yAxisLabel"])
    algorithms = list(data.keys())
    values = list(data.values())
    fig, ax = plt.subplots(figsize=PICTURE_SIZE)

    index = np.arange(len(algorithms))

    if "title" in settings:
        ax.set_title(settings["title"], fontsize="x-large")

    ax.set_xlabel("Algoritmus")
    ax.xaxis.label.set_fontsize("x-large")
    ax.set_xticks(index)
    ax.set_xticklabels(algorithms)
    for tick in ax.get_xticklabels():
        tick.set_fontsize("large")

    ax.set_ylabel(settings["yAxisLabel"])
    if len(settings["yAxisLabel"]) <= 30:
        ax.yaxis.label.set_fontsize("x-large")
    else:
        ax.yaxis.label.set_fontsize("large")

    if "yAxisFormatter" in settings:
        ax.yaxis.set_major_formatter(FuncFormatter(settings["yAxisFormatter"]))

    ax.bar(index, values, color=COLORS, edgecolor="black")

    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if "title" in settings else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))

    if show:
        plt.show()
    if not filename is None:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

# possible settings (kwargs):
# required:
#   leftYAxisLabel
#   rightYAxisLabel
# optional:
#   leftYAxisFormatter
#   rightYAxisFormatter
#   title
def plotTwinAxesSingleData(leftData, rightData, filename=None, show=False, **settings):
    _checkSettings(settings, ["leftYAxisLabel", "rightYAxisLabel"])
    leftAlgorithms = list(leftData.keys())
    leftValues = list(leftData.values())
    rightAlgorithms = list(rightData.keys())
    rightValues = list(rightData.values())
    fig, left = plt.subplots(figsize=PICTURE_SIZE)

    algorithms = leftAlgorithms + rightAlgorithms
    index = np.arange(len(algorithms))

    if "title" in settings:
        left.set_title(settings["title"], fontsize="x-large")

    left.set_xlabel("Algoritmus")
    left.xaxis.label.set_fontsize("x-large")
    left.set_xticks(index)
    left.set_xticklabels(algorithms)
    for tick in left.get_xticklabels():
        tick.set_fontsize("large")

    left.set_ylabel(settings["leftYAxisLabel"])
    if len(settings["leftYAxisLabel"]) <= 30:
        left.yaxis.label.set_fontsize("x-large")
    else:
        left.yaxis.label.set_fontsize("large")

    if "leftYAxisFormatter" in settings:
        left.yaxis.set_major_formatter(FuncFormatter(settings["leftYAxisFormatter"]))

    # assume HOG is on the first index
    left.bar(index[:len(leftAlgorithms)], leftValues, color=COLORS[:len(leftAlgorithms)], edgecolor="black")

    right = left.twinx()

    right.set_ylabel(settings["rightYAxisLabel"])
    if len(settings["rightYAxisLabel"]) <= 30:
        right.yaxis.label.set_fontsize("x-large")
    else:
        right.yaxis.label.set_fontsize("large")

    if "rightYAxisFormatter" in settings:
        right.yaxis.set_major_formatter(FuncFormatter(settings["rightYAxisFormatter"]))

    right.bar(index[len(leftAlgorithms):], rightValues, color=COLORS[len(leftAlgorithms):], edgecolor="black") # except first (which is HOG)

    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if "title" in settings else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))

    if show:
        plt.show()
    if not filename is None:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

# assume it's pairs of data, all containing same number of items

# possible settings (kwargs):
# required:
#   leftYAxisLabel
#   rightYAxisLabel
#   leftLegend
#   rightLegend
# optional:
#   leftYAxisFormatter
#   rightYAxisFormatter
#   title
def plotTwinAxesTwinData(leftData, rightData, filename=None, show=False, **settings):
    _checkSettings(settings, ["leftYAxisLabel", "rightYAxisLabel", "leftLegend", "rightLegend"])
    fig, left = plt.subplots(figsize=PICTURE_SIZE)

    right = left.twinx()

    labels = list(leftData.keys())
    leftValues = list(leftData.values())
    rightValues = list(rightData.values())

    barWidth = 0.35

    index = np.arange(len(labels))

    if "title" in settings:
        left.set_title(settings["title"], fontsize="x-large")

    left.set_xlabel("Algoritmus")
    left.xaxis.label.set_fontsize("x-large")
    left.set_xticks(index)
    left.set_xticklabels(labels)
    for tick in left.get_xticklabels():
        tick.set_fontsize("large")

    left.set_ylabel(settings["leftYAxisLabel"])
    if len(settings["leftYAxisLabel"]) <= 30:
        left.yaxis.label.set_fontsize("x-large")
    else:
        left.yaxis.label.set_fontsize("large")
    if "leftYAxisFormatter" in settings:
        left.yaxis.set_major_formatter(FuncFormatter(settings["leftYAxisFormatter"]))

    right.set_ylabel(settings["rightYAxisLabel"])
    if len(settings["rightYAxisLabel"]) <= 30:
        right.yaxis.label.set_fontsize("x-large")
    else:
        right.yaxis.label.set_fontsize("large")
    if "rightYAxisFormatter" in settings:
        right.yaxis.set_major_formatter(FuncFormatter(settings["rightYAxisFormatter"]))

    l1 = left.bar(index - barWidth / 2, leftValues, barWidth, color=COLORS, edgecolor="black")
    l2 = right.bar(index + barWidth / 2, rightValues, barWidth, color=COLORS, hatch="x", edgecolor="black")

    plt.legend([l1, l2], (settings["leftLegend"], settings["rightLegend"]))
    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if "title" in settings else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))
    if show:
        plt.show()
    if not filename is None:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

# assume data has same amount of items (same algorithms, different data)
# and none of them are empty

# possible settings (kwargs):
# required:
#   leftYAxisLabel - string
#   rightYAxisLabel - string
#   leftLegend - array of strings
#   rightLegend - array of strings
# optional:
#   leftYAxisFormatter
#   rightYAxisFormatter
#   title - string
def plotTwinAxesData(leftData, rightData, filename=None, show=False, **settings):
    # (left|right)Data are arrays of arrays of values (arrays of data sets)
    # (left|right)Legend are arrays of strings
    _checkSettings(settings, ["leftYAxisLabel", "rightYAxisLabel", "leftLegend", "rightLegend"])
    fig, left = plt.subplots(figsize=PICTURE_SIZE)

    right = left.twinx()

    mapValuesToList = lambda data: list(data.values())

    labels = list(leftData[0].keys())
    leftValues = list(map(mapValuesToList, leftData))
    rightValues = list(map(mapValuesToList, rightData))

    barWidth = 0.8 / (len(leftValues) + len(rightValues))

    index = np.arange(len(labels), dtype=np.float64)

    if "title" in settings:
        left.set_title(settings["title"], fontsize="x-large")

    left.set_xlabel("Algoritmus")
    left.xaxis.label.set_fontsize("x-large")
    left.set_xticks(index)
    left.set_xticklabels(labels)
    for tick in left.get_xticklabels():
        tick.set_fontsize("large")

    left.set_ylabel(settings["leftYAxisLabel"])
    if len(settings["leftYAxisLabel"]) <= 30:
        left.yaxis.label.set_fontsize("x-large")
    else:
        left.yaxis.label.set_fontsize("large")
    if "leftYAxisFormatter" in settings:
        left.yaxis.set_major_formatter(FuncFormatter(settings["leftYAxisFormatter"]))

    right.set_ylabel(settings["rightYAxisLabel"])
    if len(settings["rightYAxisLabel"]) <= 30:
        right.yaxis.label.set_fontsize("x-large")
    else:
        right.yaxis.label.set_fontsize("large")
    if "rightYAxisFormatter" in settings:
        right.yaxis.set_major_formatter(FuncFormatter(settings["rightYAxisFormatter"]))

    data = [
        # { "x", "y", "axis", "hatch" }
    ]

    # first, add all data on respective axis, on same "x" spots (no offsets yet)
    data += list(map(lambda data: {"x": index, "y": data, "axis": left}, leftValues))
    data += list(map(lambda data: {"x": index, "y": data, "axis": right}, rightValues))

    # second, add hatches
    for i, item in enumerate(data):
        if i == 0:
            # first data set doesn't get hatch
            item["hatch"] = None # try this
        else:
            # not safe in general, works for <= 8 datasets in total
            item["hatch"] = hatches[i - 1]

    # third, shift data on x axis
    hW = barWidth / 2

    limit = len(data) - 1
    shifts = [hW * i for i in range(-limit, len(data), 2)]
    for i, d in enumerate(data):
        d["x"] = d["x"] + shifts[i]

    colors = getColors(COLORS_HS, len(data))
    bars = []
    for i, d in enumerate(data):
        bars.append(d["axis"].bar(d["x"], d["y"], barWidth, color=colors[i], edgecolor="black"))

    plt.legend(bars, settings["leftLegend"] + settings["rightLegend"])
    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if "title" in settings else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))
    if show:
        plt.show()
    if not filename is None:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
# def testQuadrupleYAxis():
#     COLORS2 = ["r", "g", "b", "c", "m", "y", "k"]
#     def make_patch_spines_invisible(ax):
#         ax.set_frame_on(True)
#         ax.patch.set_visible(False)
#         for sp in ax.spines.values():
#             sp.set_visible(False)
#
#     fig, left1 = plt.subplots()
#
#     right1 = left1.twinx()
#     left2 = left1.twinx()
#     right2 = left1.twinx()
#
#     make_patch_spines_invisible(right1)
#     make_patch_spines_invisible(right2)
#     make_patch_spines_invisible(left2)
#
#     right2.spines["right"].set_position(("axes", 1.2))
#     left2.spines["left"].set_position(("axes", -0.2))
#
#     right2.spines["right"].set_visible(True)
#     left2.spines["left"].set_visible(True)
#     left2.yaxis.set_label_position("left")
#     left2.yaxis.set_ticks_position("left")
#
#     labels = getVariableKeys("matchingSingle")
#     matchingAll = getVariableValues("matchingAll", data.DATA_SINGLE)
#     matchingSingle = getVariableValues("matchingSingle", data.DATA_SINGLE)
#
#     barWidth = 0.35
#
#     index = np.arange(len(labels))
#
#     left1.set_title("Hledání části")
#     left1.set_xlabel("Algoritmus")
#     left1.set_xticks(index)
#     left1.set_xticklabels(labels)
#
#     left1.set_ylabel("v jednotlivém obrázku (HOG)")
#     left2.set_ylabel("ve všech obrázcích (HOG)")
#     right1.set_ylabel("v jednotlivém obrázku")
#     right2.set_ylabel("ve všech obrázcích")
#
#     left2.bar(index[:1] - barWidth/2, matchingAll[:1], barWidth, color=COLORS2[:1])
#     left1.bar(index[:1] + barWidth/2, matchingSingle[:1], barWidth, color=COLORS[:1])
#     right2.bar(index[1:] - barWidth / 2, matchingAll[1:], barWidth, color=COLORS2[1:])
#     right1.bar(index[1:] + barWidth / 2, matchingSingle[1:], barWidth, color=COLORS[1:])
#
#     tkw = dict(size=4, width=1.5)
#     left1.tick_params(axis='y', **tkw)
#     right1.tick_params(axis='y', **tkw)
#     right2.tick_params(axis='y', **tkw)
#     left2.tick_params(axis='y', **tkw)
#     left1.tick_params(axis='x', **tkw)
#
#     plt.tight_layout()
#     plt.grid(True, axis="y")
#     plt.show()

def cherryPick(dictionary, keys, include=True):
    return { key: dictionary[key] for key in dictionary.keys() if (key in keys) == include}


# plotTwinAxesData(leftData=[
#                      data.DATA_LARGE["descriptorPart"],
#                      data.DATA_640x480["descriptorPart"]
#                  ],
#                  rightData=[
#                      data.DATA_LARGE["descriptorImage"],
#                      data.DATA_640x480["descriptorImage"]
#                  ],
#                  leftYAxisLabel="Délka výpočtu deskriptoru části [ms]",
#                  rightYAxisLabel="Délka výpočtu deskriptoru obrázku [ms]",
#                  leftLegend=[
#                      "Část (300x300)",
#                      "Část (640x480)"
#                  ],
#                  rightLegend=[
#                      "Obrázek (300x300)",
#                      "Obrázek (640x480)"
#                  ],
#                  filename="test2.png")

print("Generating charts...")

plotSingleAxisSingleData(data=data.DATA_SINGLE["accuracy"],
                         yAxisLabel="Přesnost [%]",
                         yAxisFormatter=PERCENTAGE_FORMATTER,
                         filename="single/new/accuracy.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_SINGLE["matchingSingle"], ("HOG")),
                       rightData=cherryPick(data.DATA_SINGLE["matchingSingle"], ("HOG"), include=False),
                       leftYAxisLabel="Délka hledání v jednom obrázku (HOG) [ms]",
                       rightYAxisLabel="Délka hledání v jednom obrázku [ms]",
                       filename="single/new/matching.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_SINGLE["partProcess"], ("HOG")),
                       rightData=cherryPick(data.DATA_SINGLE["partProcess"], ("HOG"), include=False),
                       leftYAxisLabel="Délka zpracování celé části (HOG) [ms]",
                       rightYAxisLabel="Délka zpracování celé části [ms]",
                       filename="single/new/partProcess.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_SINGLE["totalTime"], ("HOG")),
                       rightData=cherryPick(data.DATA_SINGLE["totalTime"], ("HOG"), include=False),
                       leftYAxisLabel="Celkový čas (HOG) [ms]",
                       rightYAxisLabel="Celkový čas [ms]",
                       filename="single/new/totalTime.png")
plotTwinAxesTwinData(leftData=data.DATA_SINGLE["descriptorPart"],
                     rightData=data.DATA_SINGLE["descriptorImage"],
                     leftYAxisLabel="Délka výpočtu deskriptoru části [ms]",
                     rightYAxisLabel="Délka výpočtu deskriptoru obrázku [ms]",
                     leftLegend="Část",
                     rightLegend="Obrázek",
                     filename="single/new/descriptor.png")
plotTwinAxesTwinData(leftData=data.DATA_SINGLE["descriptorPartSize"],
                     rightData=data.DATA_SINGLE["descriptorImageSize"],
                     leftYAxisLabel="Velikost deskriptoru části",
                     rightYAxisLabel="Velikost deskriptoru obrázku",
                     leftLegend="Část",
                     rightLegend="Obrázek",
                     filename="single/new/descriptorSize.png")

print("Generating NEW charts...")

plotSingleAxisSingleData(data=data.DATA_LARGE["accuracy"],
                         yAxisLabel="Přesnost [%]",
                         yAxisFormatter=PERCENTAGE_FORMATTER,
                         filename="large/accuracy.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_LARGE["matchingSingle"], ("HOG")),
                       rightData=cherryPick(data.DATA_LARGE["matchingSingle"], ("HOG"), include=False),
                       leftYAxisLabel="Délka hledání v jednom obrázku (HOG) [ms]",
                       rightYAxisLabel="Délka hledání v jednom obrázku [ms]",
                       filename="large/matching.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_LARGE["partProcess"], ("HOG")),
                       rightData=cherryPick(data.DATA_LARGE["partProcess"], ("HOG"), include=False),
                       leftYAxisLabel="Délka zpracování celé části (HOG) [ms]",
                       rightYAxisLabel="Délka zpracování celé části [ms]",
                       filename="large/partProcess.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_LARGE["totalTime"], ("HOG")),
                       rightData=cherryPick(data.DATA_LARGE["totalTime"], ("HOG"), include=False),
                       leftYAxisLabel="Celkový čas (HOG) [s]",
                       rightYAxisLabel="Celkový čas [s]",
                       leftYAxisFormatter=SECOND_FORMATTER,
                       rightYAxisFormatter=SECOND_FORMATTER,
                       filename="large/totalTime.png")
plotTwinAxesTwinData(leftData=data.DATA_LARGE["descriptorPart"],
                     rightData=data.DATA_LARGE["descriptorImage"],
                     leftYAxisLabel="Délka výpočtu deskriptoru části [ms]",
                     rightYAxisLabel="Délka výpočtu deskriptoru obrázku [ms]",
                     leftLegend="Část",
                     rightLegend="Obrázek",
                     filename="large/descriptor.png")
plotTwinAxesTwinData(leftData=data.DATA_LARGE["descriptorPartSize"],
                     rightData=data.DATA_LARGE["descriptorImageSize"],
                     leftYAxisLabel="Velikost deskriptoru části",
                     rightYAxisLabel="Velikost deskriptoru obrázku",
                     leftLegend="Část",
                     rightLegend="Obrázek",
                     filename="large/descriptorSize.png")

print("Generating 640x480 small batch charts")


plotSingleAxisSingleData(data=data.DATA_640x480_SMALL["accuracy"],
                         yAxisLabel="Přesnost [%]",
                         yAxisFormatter=PERCENTAGE_FORMATTER,
                         filename="640x480_small/accuracy.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_640x480_SMALL["matchingSingle"], ("HOG")),
                       rightData=cherryPick(data.DATA_640x480_SMALL["matchingSingle"], ("HOG"), include=False),
                       leftYAxisLabel="Délka hledání v jednom obrázku (HOG) [ms]",
                       rightYAxisLabel="Délka hledání v jednom obrázku [ms]",
                       filename="640x480_small/matching.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_640x480_SMALL["partProcess"], ("HOG")),
                       rightData=cherryPick(data.DATA_640x480_SMALL["partProcess"], ("HOG"), include=False),
                       leftYAxisLabel="Délka zpracování celé části (HOG) [ms]",
                       rightYAxisLabel="Délka zpracování celé části [ms]",
                       filename="640x480_small/partProcess.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_640x480_SMALL["totalTime"], ("HOG")),
                       rightData=cherryPick(data.DATA_640x480_SMALL["totalTime"], ("HOG"), include=False),
                       leftYAxisLabel="Celkový čas (HOG) [s]",
                       rightYAxisLabel="Celkový čas [s]",
                       leftYAxisFormatter=SECOND_FORMATTER,
                       rightYAxisFormatter=SECOND_FORMATTER,
                       filename="640x480_small/totalTime.png")
plotTwinAxesTwinData(leftData=data.DATA_640x480_SMALL["descriptorPart"],
                     rightData=data.DATA_640x480_SMALL["descriptorImage"],
                     leftYAxisLabel="Délka výpočtu deskriptoru části [ms]",
                     rightYAxisLabel="Délka výpočtu deskriptoru obrázku [ms]",
                     leftLegend="Část",
                     rightLegend="Obrázek",
                     filename="640x480_small/descriptor.png")
plotTwinAxesTwinData(leftData=data.DATA_640x480_SMALL["descriptorPartSize"],
                     rightData=data.DATA_640x480_SMALL["descriptorImageSize"],
                     leftYAxisLabel="Velikost deskriptoru části",
                     rightYAxisLabel="Velikost deskriptoru obrázku",
                     leftLegend="Část",
                     rightLegend="Obrázek",
                     filename="640x480_small/descriptorSize.png")

print("Generating 640x480 charts")

plotSingleAxisSingleData(data=data.DATA_640x480["accuracy"],
                         yAxisLabel="Přesnost [%]",
                         yAxisFormatter=PERCENTAGE_FORMATTER,
                         filename="640x480_large/accuracy.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_640x480["matchingSingle"], ("HOG")),
                       rightData=cherryPick(data.DATA_640x480["matchingSingle"], ("HOG"), include=False),
                       leftYAxisLabel="Délka hledání v jednom obrázku (HOG) [ms]",
                       rightYAxisLabel="Délka hledání v jednom obrázku [ms]",
                       filename="640x480_large/matching.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_640x480["partProcess"], ("HOG")),
                       rightData=cherryPick(data.DATA_640x480["partProcess"], ("HOG"), include=False),
                       leftYAxisLabel="Délka zpracování celé části (HOG) [ms]",
                       rightYAxisLabel="Délka zpracování celé části [ms]",
                       filename="640x480_large/partProcess.png")
plotTwinAxesSingleData(leftData=cherryPick(data.DATA_640x480["totalTime"], ("HOG")),
                       rightData=cherryPick(data.DATA_640x480["totalTime"], ("HOG"), include=False),
                       leftYAxisLabel="Celkový čas (HOG) [s]",
                       rightYAxisLabel="Celkový čas [s]",
                       leftYAxisFormatter=SECOND_FORMATTER,
                       rightYAxisFormatter=SECOND_FORMATTER,
                       filename="640x480_large/totalTime.png")
plotTwinAxesTwinData(leftData=data.DATA_640x480["descriptorPart"],
                     rightData=data.DATA_640x480["descriptorImage"],
                     leftYAxisLabel="Délka výpočtu deskriptoru části [ms]",
                     rightYAxisLabel="Délka výpočtu deskriptoru obrázku [ms]",
                     leftLegend="Část",
                     rightLegend="Obrázek",
                     filename="640x480_large/descriptor.png")
plotTwinAxesTwinData(leftData=data.DATA_640x480["descriptorPartSize"],
                     rightData=data.DATA_640x480["descriptorImageSize"],
                     leftYAxisLabel="Velikost deskriptoru části",
                     rightYAxisLabel="Velikost deskriptoru obrázku",
                     leftLegend="Část",
                     rightLegend="Obrázek",
                     filename="640x480_large/descriptorSize.png")
print("Done!")