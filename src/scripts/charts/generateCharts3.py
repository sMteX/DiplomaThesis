import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import src.scripts.charts.data as data

OUTPUT_DIR = "./output/allSizes"

# slightly lighter
COLORS_300x300 = ["#ff23ff", "#ff4c4c", "#ff9e3d", "#ffe44c", "#00e500", "#47a0ff", "#a347ff"]
COLORS_640x480 = ["#cc00cc", "#e00000", "#e07000", "#efcf00", "#009e02", "#006ce0", "#7300e0"]
#slightly darker
COLORS_1280x720 = ["#990099", "#990000", "#994c00", "#b29700", "#006600", "#004c9e", "#4c0099"]

PERCENTAGE_FORMATTER = lambda y, _: f"{(100 * y):.0f} %"
SECOND_FORMATTER = lambda y, _: math.floor(y / 1000.0)
DEFAULT_LEGEND = [
    patches.Patch(edgecolor="black", facecolor=COLORS_300x300[0], label="300x300"),
    patches.Patch(edgecolor="black", facecolor=COLORS_640x480[0], label="640x480"),
    patches.Patch(edgecolor="black", facecolor=COLORS_1280x720[0], label="1280x720")
]
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

def cherryPick(dictionary, keys, include=True):
    return { key: dictionary[key] for key in dictionary.keys() if (key in keys) == include}

def pickColors(colors, indexes):
    return [colors[i] for i in indexes]

algorithmMap = {
    "HOG": 0,
    "FT": 1,
    "SIFT": 2,
    "SURF": 3,
    "BRIEF": 4,
    "ORB": 5,
    "FREAK": 6
}

def splitIntoXY(data):
    # takes dictionary of alg: value, returns tuple (xValues, yValues) where xValues stays constant for same algorithm regardless of ordering
    keys = list(data.keys())
    y = list(data.values())
    x = list(map(lambda key: algorithmMap[key], keys))
    return np.asarray(x), np.asarray(y)

def accuracy(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_LARGE["accuracy"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["accuracy"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["accuracy"])
    
    fig, axis = plt.subplots(figsize=PICTURE_SIZE)
    
    if title:
        axis.set_title("Přesnost algoritmů", fontsize="x-large")

    axis.set_xlabel("Algoritmus")
    axis.xaxis.label.set_fontsize("x-large")
    axis.set_xticks(list(algorithmMap.values()))
    axis.set_xticklabels(list(algorithmMap.keys()))
    for tick in axis.get_xticklabels():
        tick.set_fontsize("large")

    axis.set_ylabel("Přesnost [%]")
    axis.yaxis.label.set_fontsize("x-large")

    axis.yaxis.set_major_formatter(FuncFormatter(PERCENTAGE_FORMATTER))

    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    axis.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    axis.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    axis.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    plt.legend(handles=DEFAULT_LEGEND)
    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))

    if show:
        plt.show()
    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

def partDescriptorTime(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_LARGE["descriptorPart"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["descriptorPart"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["descriptorPart"])

    fig, axis = plt.subplots(figsize=PICTURE_SIZE)

    if title:
        axis.set_title("Délka výpočtu deskriptoru části", fontsize="x-large")

    axis.set_xlabel("Algoritmus")
    axis.xaxis.label.set_fontsize("x-large")
    axis.set_xticks(list(algorithmMap.values()))
    axis.set_xticklabels(list(algorithmMap.keys()))
    for tick in axis.get_xticklabels():
        tick.set_fontsize("large")

    axis.set_ylabel("Délka výpočtu deskriptoru části [ms]")
    axis.yaxis.label.set_fontsize("x-large")

    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    axis.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    axis.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    axis.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    plt.legend(handles=DEFAULT_LEGEND)
    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))

    if show:
        plt.show()
    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

def imageDescriptorTime(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_LARGE["descriptorImage"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["descriptorImage"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["descriptorImage"])

    fig, axis = plt.subplots(figsize=PICTURE_SIZE)

    if title:
        axis.set_title("Délka výpočtu deskriptoru obrázku", fontsize="x-large")

    axis.set_xlabel("Algoritmus")
    axis.xaxis.label.set_fontsize("x-large")
    axis.set_xticks(list(algorithmMap.values()))
    axis.set_xticklabels(list(algorithmMap.keys()))
    for tick in axis.get_xticklabels():
        tick.set_fontsize("large")

    axis.set_ylabel("Délka výpočtu deskriptoru obrázku [ms]")
    axis.yaxis.label.set_fontsize("x-large")

    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    axis.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    axis.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    axis.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    plt.legend(handles=DEFAULT_LEGEND)
    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))

    if show:
        plt.show()
    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))


def partDescriptorSize(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_LARGE["descriptorPartSize"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["descriptorPartSize"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["descriptorPartSize"])

    fig, axis = plt.subplots(figsize=PICTURE_SIZE)

    if title:
        axis.set_title("Průměrná velikost deskriptoru části", fontsize="x-large")

    axis.set_xlabel("Algoritmus")
    axis.xaxis.label.set_fontsize("x-large")
    axis.set_xticks(list(algorithmMap.values()))
    axis.set_xticklabels(list(algorithmMap.keys()))
    for tick in axis.get_xticklabels():
        tick.set_fontsize("large")

    axis.set_ylabel("Průměrná velikost deskriptoru části")
    axis.yaxis.label.set_fontsize("x-large")

    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    axis.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    axis.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    axis.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    plt.legend(handles=DEFAULT_LEGEND)
    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))

    if show:
        plt.show()
    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

def imageDescriptorSize(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_LARGE["descriptorImageSize"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["descriptorImageSize"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["descriptorImageSize"])

    fig, axis = plt.subplots(figsize=PICTURE_SIZE)

    if title:
        axis.set_title("Průměrná velikost deskriptoru obrázku", fontsize="x-large")

    axis.set_xlabel("Algoritmus")
    axis.xaxis.label.set_fontsize("x-large")
    axis.set_xticks(list(algorithmMap.values()))
    axis.set_xticklabels(list(algorithmMap.keys()))
    for tick in axis.get_xticklabels():
        tick.set_fontsize("large")

    axis.set_ylabel("Průměrná velikost deskriptoru obrázku")
    axis.yaxis.label.set_fontsize("x-large")

    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    axis.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    axis.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    axis.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    plt.legend(handles=DEFAULT_LEGEND)
    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))

    if show:
        plt.show()
    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

def matching(title=False, filename=None, show=False):
    hogSmallX, hogSmallY = splitIntoXY(cherryPick(data.DATA_LARGE["matchingSingle"], ["HOG"]))
    hogMediumX, hogMediumY = splitIntoXY(cherryPick(data.DATA_640x480["matchingSingle"], ["HOG"]))
    smallX, smallY = splitIntoXY(cherryPick(data.DATA_LARGE["matchingSingle"], ["HOG"], include=False))
    mediumX, mediumY = splitIntoXY(cherryPick(data.DATA_640x480["matchingSingle"], ["HOG"], include=False))
    largeX, largeY = splitIntoXY(cherryPick(data.DATA_1280x720["matchingSingle"], ["HOG"], include=False))

    fig, left = plt.subplots(figsize=PICTURE_SIZE)
    right = left.twinx()

    if title:
        left.set_title("Délka hledání v jednom obrázku", fontsize="x-large")

    left.set_xlabel("Algoritmus")
    left.xaxis.label.set_fontsize("x-large")
    left.set_xticks(list(algorithmMap.values()))
    left.set_xticklabels(list(algorithmMap.keys()))
    for tick in left.get_xticklabels():
        tick.set_fontsize("large")

    left.set_ylabel("Délka hledání v jednom obrázku (HOG) [ms]")
    left.yaxis.label.set_fontsize("x-large")

    right.set_ylabel("Délka hledání v jednom obrázku [ms]")
    right.yaxis.label.set_fontsize("x-large")

    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    left.bar(hogSmallX - 2 * hW, hogSmallY, barWidth, color=pickColors(COLORS_300x300, hogSmallX), edgecolor="black")
    left.bar(hogMediumX, hogMediumY, barWidth, color=pickColors(COLORS_640x480, hogMediumX), edgecolor="black")
    right.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    right.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    right.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    plt.legend(handles=DEFAULT_LEGEND)
    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))

    if show:
        plt.show()
    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

def partProcess(title=False, filename=None, show=False):
    hogSmallX, hogSmallY = splitIntoXY(cherryPick(data.DATA_LARGE["partProcess"], ["HOG"]))
    hogMediumX, hogMediumY = splitIntoXY(cherryPick(data.DATA_640x480["partProcess"], ["HOG"]))
    smallX, smallY = splitIntoXY(cherryPick(data.DATA_LARGE["partProcess"], ["HOG"], include=False))
    mediumX, mediumY = splitIntoXY(cherryPick(data.DATA_640x480["partProcess"], ["HOG"], include=False))
    largeX, largeY = splitIntoXY(cherryPick(data.DATA_1280x720["partProcess"], ["HOG"], include=False))

    fig, left = plt.subplots(figsize=PICTURE_SIZE)
    right = left.twinx()

    if title:
        left.set_title("Délka zpracování celé části", fontsize="x-large")

    left.set_xlabel("Algoritmus")
    left.xaxis.label.set_fontsize("x-large")
    left.set_xticks(list(algorithmMap.values()))
    left.set_xticklabels(list(algorithmMap.keys()))
    for tick in left.get_xticklabels():
        tick.set_fontsize("large")

    left.set_ylabel("Délka zpracování celé části (HOG) [ms]")
    left.yaxis.label.set_fontsize("x-large")

    right.set_ylabel("Délka zpracování celé části [ms]")
    right.yaxis.label.set_fontsize("x-large")

    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    left.bar(hogSmallX - 2 * hW, hogSmallY, barWidth, color=pickColors(COLORS_300x300, hogSmallX), edgecolor="black")
    left.bar(hogMediumX, hogMediumY, barWidth, color=pickColors(COLORS_640x480, hogMediumX), edgecolor="black")
    right.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    right.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    right.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    plt.legend(handles=DEFAULT_LEGEND)
    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))

    if show:
        plt.show()
    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

def totalTime(title=False, filename=None, show=False):
    # def make_patch_spines_invisible(ax):
    #     ax.set_frame_on(True)
    #     ax.patch.set_visible(False)
    #     for sp in ax.spines.values():
    #         sp.set_visible(False)

    hogSmallX, hogSmallY = splitIntoXY(cherryPick(data.DATA_LARGE["totalTime"], ["HOG"]))
    hogMediumX, hogMediumY = splitIntoXY(cherryPick(data.DATA_640x480["totalTime"], ["HOG"]))
    smallX, smallY = splitIntoXY(cherryPick(data.DATA_LARGE["totalTime"], ["HOG"], include=False))
    mediumX, mediumY = splitIntoXY(cherryPick(data.DATA_640x480["totalTime"], ["HOG"], include=False))
    largeX, largeY = splitIntoXY(cherryPick(data.DATA_1280x720["totalTime"], ["HOG"], include=False))

    fig, (upper, bottom) = plt.subplots(2, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Délka zpracování celé části", fontsize="x-large")

    fig.text(0.07, 0.55, "Délka zpracování celé části [s]", va="center", rotation="vertical", fontsize="x-large")
    bottom.yaxis.label.set_fontsize("x-large")

    bottom.set_xlabel("Algoritmus")
    bottom.xaxis.label.set_fontsize("x-large")
    bottom.set_xticks(list(algorithmMap.values()))
    bottom.set_xticklabels(list(algorithmMap.keys()))
    for tick in bottom.get_xticklabels():
        tick.set_fontsize("large")

    bottom.yaxis.set_major_formatter(FuncFormatter(SECOND_FORMATTER))
    upper.yaxis.set_major_formatter(FuncFormatter(SECOND_FORMATTER))

    bottom.set_ylim(0, 150000)
    upper.set_ylim(150000, 2600000)
    upper.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    bottom.bar(hogSmallX - 2 * hW, hogSmallY, barWidth, color=pickColors(COLORS_300x300, hogSmallX), edgecolor="black")
    upper.bar(hogSmallX - 2 * hW, hogSmallY, barWidth, color=pickColors(COLORS_300x300, hogSmallX), edgecolor="black")
    bottom.bar(hogMediumX, hogMediumY, barWidth, color=pickColors(COLORS_640x480, hogMediumX), edgecolor="black")
    upper.bar(hogMediumX, hogMediumY, barWidth, color=pickColors(COLORS_640x480, hogMediumX), edgecolor="black")
    bottom.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    bottom.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    bottom.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")
    upper.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    upper.legend(handles=DEFAULT_LEGEND)
    bottom.grid(True, axis="y")
    upper.grid(True, axis="y")

    d = 0.01
    kwargs = dict(transform=bottom.transAxes, color="black", clip_on=False)
    bottom.plot((-d, d),(1 - d, 1 + d), **kwargs)
    bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=upper.transAxes)
    upper.plot((-d, d), (-d, d), **kwargs)
    upper.plot((1 - d, 1 + d), (-d, d), **kwargs)

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.1)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

# accuracy(show=True)
# partDescriptorTime(show=True)
# imageDescriptorTime(show=True)
# partDescriptorSize(show=True)
# imageDescriptorSize(show=True)
# matching(show=True)
# partProcess(show=True)
totalTime(filename="totalTime.png")