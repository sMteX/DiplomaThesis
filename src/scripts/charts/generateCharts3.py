import os
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.interpolate import griddata
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import src.scripts.charts.data as data

OUTPUT_DIR = "./output/allSizesNewLargerHOG/gray/subplotCount"
# global default font size, font sizes like xx-large are relative to this
matplotlib.rcParams['font.size'] = 14   # default = 10

Color = namedtuple("Color", "light, normal, dark")
GRAY = Color(light="#b5b5b5", normal="#878787", dark="#444444")
# slightly lighter
COLORS_300x300 = ["#ff23ff", "#ff4c4c", "#ff9e3d", "#ffe44c", "#00e500", "#47a0ff", "#a347ff"]
COLORS_640x480 = ["#cc00cc", "#e00000", "#e07000", "#efcf00", "#009e02", "#006ce0", "#7300e0"]
#slightly darker
COLORS_1280x720 = ["#990099", "#990000", "#994c00", "#b29700", "#006600", "#004c9e", "#4c0099"]

PERCENTAGE_FORMATTER = lambda y, _: f"{(100 * y):.0f} %"
SECOND_THOUSAND_FORMATTER = lambda y, _: math.floor(y / 1000.0)
def DECIMAL_SECOND_FORMATTER(y, _):
    if y < 1000:
        return f"{(y / 1000.0):.1f}"
    else:
        return f"{(y / 1000.0):.0f}"

DEFAULT_LEGEND = [
    patches.Patch(edgecolor="black", facecolor=COLORS_300x300[0], label="300x300"),
    patches.Patch(edgecolor="black", facecolor=COLORS_640x480[0], label="640x480"),
    patches.Patch(edgecolor="black", facecolor=COLORS_1280x720[0], label="1280x720")
]
GRAY_LEGEND = [
    patches.Patch(edgecolor="black", facecolor=GRAY.light, label="300x300"),
    patches.Patch(edgecolor="black", facecolor=GRAY.normal, label="640x480"),
    patches.Patch(edgecolor="black", facecolor=GRAY.dark, label="1280x720")
]
PICTURE_SIZE = (12.5, 10)
DEFAULT_MARGINS = {
    "left": 2.3,
    "right": 2.3,
    "bottom": 1
}
TOP_MARGIN_NO_TITLE = 0.2
TOP_MARGIN_TITLE = 0.3

def transformMargins(left, right, top, bottom, pictureSize):
    return {
        "left": left / pictureSize[0],
        "right": 1 - right / pictureSize[0],
        "top": 1 - top / pictureSize[1],
        "bottom": bottom / pictureSize[1]
    }

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

def setupXAxis(axis):
    axis.set_xlabel("Algoritmus")
    axis.xaxis.label.set_fontsize("xx-large")
    axis.xaxis.labelpad = 12    # default = 4
    axis.set_xticks(list(algorithmMap.values()))
    axis.set_xticklabels(list(algorithmMap.keys()))
    for tick in axis.get_xticklabels():
        tick.set_fontsize("x-large")

def drawAxisSplitters(*axes, size=0.01):
    if len(axes) < 2:
        raise ValueError("Need at least 2 axes")
    bottom, *middle, upper = axes

    kwargs = dict(transform=bottom.transAxes, color="black", clip_on=False)
    bottom.plot((-size, size), (1 - size, 1 + size), **kwargs)
    bottom.plot((1 - size, 1 + size), (1 - size, 1 + size), **kwargs)
    
    for mid in middle:
        kwargs.update(transform=mid.transAxes)
        mid.plot((-size, size), (1 - size, 1 + size), **kwargs)
        mid.plot((1 - size, 1 + size), (1 - size, 1 + size), **kwargs)
        mid.plot((-size, size), (-size, size), **kwargs)
        mid.plot((1 - size, 1 + size), (-size, size), **kwargs)
    
    kwargs.update(transform=upper.transAxes)
    upper.plot((-size, size), (-size, size), **kwargs)
    upper.plot((1 - size, 1 + size), (-size, size), **kwargs)

def drawAcross(axes, *args, **kwargs):
    for axis in axes:
        axis.bar(*args, **kwargs)

def accuracy(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["accuracy"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["accuracy"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["accuracy"])
    
    fig, axis = plt.subplots(figsize=PICTURE_SIZE)
    
    if title:
        axis.set_title("Přesnost algoritmů", fontsize="xx-large")

    setupXAxis(axis)

    axis.set_ylabel("Přesnost [%]")
    axis.yaxis.label.set_fontsize("xx-large")
    for tick in axis.get_yticklabels():
        tick.set_fontsize("x-large")
    axis.yaxis.set_major_formatter(FuncFormatter(PERCENTAGE_FORMATTER))
    axis.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    axis.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    axis.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    axis.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), handles=GRAY_LEGEND, fontsize='large')
    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS))

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def partDescriptorTime(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["descriptorPart"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["descriptorPart"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["descriptorPart"])

    fig, axis = plt.subplots(figsize=PICTURE_SIZE)

    if title:
        axis.set_title("Délka výpočtu deskriptoru části", fontsize="xx-large")

    setupXAxis(axis)

    axis.set_ylabel("Délka výpočtu deskriptoru části [ms]")
    axis.yaxis.label.set_fontsize("xx-large")
    for tick in axis.get_yticklabels():
        tick.set_fontsize("x-large")
    axis.set_ylim(0, 30)
    axis.yaxis.set_major_locator(plt.MultipleLocator(2.5))
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    axis.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    axis.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    axis.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    plt.legend(loc="center left", bbox_to_anchor=(1.0125, 0.5), handles=GRAY_LEGEND, fontsize='large')
    plt.grid(True, axis="y")

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.15)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def imageDescriptorTime(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["descriptorImage"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["descriptorImage"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["descriptorImage"])

    fig, (upper, bottom) = plt.subplots(2, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Délka výpočtu deskriptoru obrázku", fontsize="xx-large")

    setupXAxis(bottom)

    fig.text(0.07, 0.55, "Délka výpočtu deskriptoru obrázku [ms]", va="center", rotation="vertical", fontsize="xx-large")

    for axis in [bottom, upper]:
        for tick in axis.get_yticklabels():
            tick.set_fontsize("x-large")

    bottom.set_ylim(0, 85)
    upper.set_ylim(100, 420)
    upper.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    bottom.yaxis.set_major_locator(plt.LinearLocator(8))
    upper.yaxis.set_major_locator(plt.LinearLocator(8))

    bottom.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, upper], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    upper.legend(loc="center left", bbox_to_anchor=(1.0125, -0.1), handles=GRAY_LEGEND, fontsize='large')
    bottom.grid(True, axis="y")
    upper.grid(True, axis="y")

    drawAxisSplitters(bottom, upper)

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.15)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def partDescriptorSize(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["descriptorPartSize"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["descriptorPartSize"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["descriptorPartSize"])

    fig, (top, upper, middle, bottom) = plt.subplots(4, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Průměrná velikost deskriptoru části", fontsize="xx-large")

    fig.text(0.06, 0.55, "Průměrná velikost deskriptoru části (v tisících)", va="center", rotation="vertical", fontsize="xx-large")

    setupXAxis(bottom)

    for axis in [top, bottom, middle, upper]:
        for tick in axis.get_yticklabels():
            tick.set_fontsize("x-large")

    bottom.set_ylim(0, 8000)
    middle.set_ylim(10000, 30000)
    middle.xaxis.tick_top()
    upper.set_ylim(43000, 57000)
    upper.xaxis.tick_top()
    top.set_ylim(100000, 105000)
    top.xaxis.tick_top()

    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    bottom.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))
    middle.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))
    upper.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))
    top.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))

    bottom.yaxis.set_major_locator(plt.LinearLocator(5))
    middle.yaxis.set_major_locator(plt.LinearLocator(5))
    upper.yaxis.set_major_locator(plt.LinearLocator(5))
    top.yaxis.set_major_locator(plt.LinearLocator(5))

    drawAcross([bottom, middle], smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, middle, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, middle, upper, top], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    middle.legend(loc="center left", bbox_to_anchor=(1.0125, 0.5), handles=GRAY_LEGEND, fontsize='large')
    bottom.grid(True, axis="y")
    middle.grid(True, axis="y")
    upper.grid(True, axis="y")
    top.grid(True, axis="y")

    drawAxisSplitters(bottom, middle, upper, top)

    topMargin = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=topMargin, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.25)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def imageDescriptorSize(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["descriptorImageSize"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["descriptorImageSize"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["descriptorImageSize"])

    fig, (top, upper, middle, bottom) = plt.subplots(4, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Průměrná velikost deskriptoru obrázku", fontsize="xx-large")

    setupXAxis(bottom)

    fig.text(0.04, 0.55, "Průměrná velikost deskriptoru obrázku (v tisících)", va="center", rotation="vertical", fontsize="xx-large")

    for axis in [bottom, middle, upper, top]:
        for tick in axis.get_yticklabels():
            tick.set_fontsize("x-large")

    bottom.set_ylim(0, 220000)
    middle.set_ylim(300000, 500000)
    middle.xaxis.tick_top()
    upper.set_ylim(600000, 900000)
    upper.xaxis.tick_top()
    top.set_ylim(2000000, 2100000)
    top.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    bottom.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))
    middle.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))
    upper.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))
    top.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))
    bottom.yaxis.set_major_locator(plt.LinearLocator(5))
    middle.yaxis.set_major_locator(plt.LinearLocator(5))
    upper.yaxis.set_major_locator(plt.LinearLocator(5))
    top.yaxis.set_major_locator(plt.LinearLocator(5))

    drawAcross([bottom, middle], smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, middle, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, middle, upper, top], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    middle.legend(loc="center left", bbox_to_anchor=(1.0125, -0.1), handles=GRAY_LEGEND, fontsize='large')
    bottom.grid(True, axis="y")
    middle.grid(True, axis="y")
    upper.grid(True, axis="y")
    top.grid(True, axis="y")

    drawAxisSplitters(bottom, middle, upper, top)

    topMargin = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=topMargin, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.25)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def matching(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["matchingSingle"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["matchingSingle"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["matchingSingle"])

    fig, (top, upper, middle, bottom) = plt.subplots(4, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Délka hledání v jednom obrázku", fontsize="xx-large")

    setupXAxis(bottom)

    fig.text(0.07, 0.55, "Délka hledání v jednom obrázku [ms]", va="center", rotation="vertical", fontsize="xx-large")

    for axis in [bottom, middle, upper, top]:
        for tick in axis.get_yticklabels():
            tick.set_fontsize("x-large")

    bottom.set_ylim(0, 40)
    middle.set_ylim(80, 140)
    middle.xaxis.tick_top()
    upper.set_ylim(600, 700)
    upper.xaxis.tick_top()
    top.set_ylim(4500, 5500)
    top.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    bottom.yaxis.set_major_locator(plt.LinearLocator(5))
    middle.yaxis.set_major_locator(plt.LinearLocator(5))
    upper.yaxis.set_major_locator(plt.LinearLocator(5))
    top.yaxis.set_major_locator(plt.LinearLocator(5))

    drawAcross([bottom, middle], smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, middle, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, middle, upper, top], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    middle.legend(loc="center left", bbox_to_anchor=(1.0125, 0.5), handles=GRAY_LEGEND, fontsize='large')
    bottom.grid(True, axis="y")
    middle.grid(True, axis="y")
    upper.grid(True, axis="y")
    top.grid(True, axis="y")

    drawAxisSplitters(bottom, middle, upper, top)

    topMargin = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=topMargin, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.25)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def partProcess(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["partProcess"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["partProcess"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["partProcess"])

    fig, (top, upper, middle, bottom) = plt.subplots(4, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Délka zpracování celé části", fontsize="xx-large")

    setupXAxis(bottom)

    fig.text(0.06, 0.55, "Délka zpracování celé části [s]", va="center", rotation="vertical", fontsize="xx-large")

    for axis in [bottom, middle, upper, top]:
        for tick in axis.get_yticklabels():
            tick.set_fontsize("x-large")

    bottom.set_ylim(0, 600)
    middle.set_ylim(1000, 11000)
    middle.xaxis.tick_top()
    upper.set_ylim(46000, 52000)
    upper.xaxis.tick_top()
    top.set_ylim(350000, 400000)
    top.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    bottom.yaxis.set_major_formatter(FuncFormatter(DECIMAL_SECOND_FORMATTER))
    middle.yaxis.set_major_formatter(FuncFormatter(DECIMAL_SECOND_FORMATTER))
    upper.yaxis.set_major_formatter(FuncFormatter(DECIMAL_SECOND_FORMATTER))
    top.yaxis.set_major_formatter(FuncFormatter(DECIMAL_SECOND_FORMATTER))

    bottom.yaxis.set_major_locator(plt.LinearLocator(5))
    middle.yaxis.set_major_locator(plt.LinearLocator(5))
    upper.yaxis.set_major_locator(plt.LinearLocator(5))
    top.yaxis.set_major_locator(plt.LinearLocator(5))

    drawAcross([bottom, middle], smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, middle, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, middle, upper, top], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    middle.legend(loc="center left", bbox_to_anchor=(1.0125, 0.5), handles=GRAY_LEGEND, fontsize='large')
    bottom.grid(True, axis="y")
    middle.grid(True, axis="y")
    upper.grid(True, axis="y")
    top.grid(True, axis="y")

    drawAxisSplitters(bottom, middle, upper, top)

    topMargin = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=topMargin, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.25)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def totalTime(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["totalTime"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["totalTime"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["totalTime"])

    fig, (top, upper, middle, bottom) = plt.subplots(4, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Celkový čas", fontsize="xx-large")

    fig.text(0.07, 0.55, "Celkový čas [s]", va="center", rotation="vertical", fontsize="xx-large")

    setupXAxis(bottom)

    for axis in [bottom, middle, upper, top]:
        for tick in axis.get_yticklabels():
            tick.set_fontsize("x-large")

    bottom.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))
    middle.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))
    upper.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))
    top.yaxis.set_major_formatter(FuncFormatter(SECOND_THOUSAND_FORMATTER))

    bottom.yaxis.set_major_locator(plt.LinearLocator(5))
    middle.yaxis.set_major_locator(plt.LinearLocator(5))
    upper.yaxis.set_major_locator(plt.LinearLocator(5))
    top.yaxis.set_major_locator(plt.LinearLocator(5))

    bottom.set_ylim(0, 30000)
    middle.set_ylim(60000, 130000)
    middle.xaxis.tick_top()
    upper.set_ylim(250000, 2600000)
    upper.xaxis.tick_top()
    top.set_ylim(18000000, 19000000)
    top.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    drawAcross([bottom, middle, upper], smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, middle, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, middle, upper, top], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    middle.legend(loc="center left", bbox_to_anchor=(1.0125, 0.5), handles=GRAY_LEGEND, fontsize='large')
    bottom.grid(True, axis="y")
    middle.grid(True, axis="y")
    upper.grid(True, axis="y")
    top.grid(True, axis="y")

    drawAxisSplitters(bottom, middle, upper, top)

    topMargin = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=topMargin, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.25)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def lighting(data, annotate=False, title=None, filename=None, show=False):
    def getData(data):
        x = []
        y = []
        z = []
        for b, val in data.items():
            for c, acc in val.items():
                x.append(b)
                y.append(c)
                z.append(acc)
        return np.asarray(x), np.asarray(y), np.asarray(z)

    colorMap = plt.cm.get_cmap("RdYlGn")
    x, y, z = getData(data)

    fig, axis = plt.subplots(figsize=PICTURE_SIZE)

    axis.set_ylabel("Kontrast", fontsize="x-large")
    axis.set_xlabel("Jas", fontsize="x-large")
    axis.set_ylim([-90, 90])
    axis.set_xlim([-90, 90])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f} %"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f} %"))

    for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick.set_fontsize("large")

    sc = axis.scatter(x, y, c=z, vmin=0, vmax=1, cmap=colorMap)
    fig.colorbar(sc, format=FuncFormatter(PERCENTAGE_FORMATTER))

    if annotate:
        for i in range(len(x)):
            _x = x[i]
            _y = y[i]
            acc = z[i]
            axis.annotate(PERCENTAGE_FORMATTER(acc, 0), (_x, _y), textcoords="offset pixels", xytext=(-10, 13))

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(left=2.2, right=0.8, bottom=0.6, top=top, pictureSize=PICTURE_SIZE), hspace=0.15)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def lightingContour(data, colormap="viridis", interpolated=False, showOriginal=False, title=None, filename=None, show=False):
    def getData(data):
        x = []
        y = []
        z = []
        for b, val in data.items():
            for c, acc in val.items():
                x.append(b)
                y.append(c)
                z.append(acc)
        return np.asarray(x), np.asarray(y), np.asarray(z)

    colorMap = plt.cm.get_cmap(colormap)

    fig, axis = plt.subplots(figsize=PICTURE_SIZE)

    axis.set_ylabel("Kontrast", fontsize="xx-large")
    axis.set_xlabel("Jas", fontsize="xx-large")
    axis.xaxis.labelpad = 12

    formatter = lambda y, _: f"{y:.0f} %" if y < 0 else (f"{y:.0f} %" if y == 0 else f"+{y:.0f} %")
    axis.xaxis.set_major_formatter(FuncFormatter(formatter))
    axis.yaxis.set_major_formatter(FuncFormatter(formatter))

    axis.tick_params(pad=10)
    for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick.set_fontsize("x-large")

    if interpolated:
        points = []
        values = []
        for b, vals in data.items():
            for c, value in vals.items():
                points.append((b, c))
                values.append(value)

        grid_x, grid_y = np.mgrid[-80:80:80j, -80:80:80j]
        grid = griddata(np.asarray(points, dtype=np.float), np.asarray(values, dtype=np.float), (grid_x, grid_y), method='cubic')
        grid = np.clip(grid, 0, 1)
        cont = axis.contourf(grid_x, grid_y, grid, levels=10, cmap=colorMap)
        colorBar = fig.colorbar(cont, format=FuncFormatter(PERCENTAGE_FORMATTER))
        colorBar.set_label("Přesnost [%]", fontsize="x-large")
    else:
        X = np.asarray(list(data.keys()))
        Y = np.asarray(list(data[0].keys()))
        Z = np.empty((len(Y), len(X)), dtype=np.float)

        i = 0
        for b, val in data.items():
            j = 0
            for c, a in val.items():
                Z[j, i] = a
                j += 1
            i += 1

        cont = axis.contourf(X, Y, Z, levels=10, cmap=colorMap)
        colorBar = fig.colorbar(cont, format=FuncFormatter(PERCENTAGE_FORMATTER))
        colorBar.set_label("Přesnost [%]", fontsize="xx-large")
        colorBar.ax.tick_params(labelsize="x-large")


    if showOriginal:
        _x, _y, _z = getData(data)
        axis.scatter(_x, _y, c=_z, cmap=colorMap, edgecolor="black")

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.grid(True, color="black", alpha=0.75)
    plt.subplots_adjust(**transformMargins(left=2, right=0.4, bottom=1, top=top, pictureSize=PICTURE_SIZE), hspace=0.15)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

accuracy(filename="1_accuracy.png")
# partDescriptorTime(filename="partDescriptorTime.png")
imageDescriptorTime(filename="2_imageDescriptorTime.png")
# partDescriptorSize(filename="partDescriptorSize.png")
imageDescriptorSize(filename="4_imageDescriptorSize.png")
# matching(filename="matching.png")
# partProcess(filename="partProcess.png")
# totalTime(filename="totalTime.png")
# # lighting(data=data.DATA_LIGHTING_FT, show=True)
# lightingContour(data=data.DATA_LIGHTING_FT, colormap="plasma", filename="ftLighting_plasma.png")