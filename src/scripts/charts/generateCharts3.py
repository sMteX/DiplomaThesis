import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import src.scripts.charts.data as data

OUTPUT_DIR = "./output/allSizesNew"

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
    axis.xaxis.label.set_fontsize("x-large")
    axis.set_xticks(list(algorithmMap.values()))
    axis.set_xticklabels(list(algorithmMap.keys()))
    for tick in axis.get_xticklabels():
        tick.set_fontsize("large")

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
        axis.set_title("Přesnost algoritmů", fontsize="x-large")

    setupXAxis(axis)

    axis.set_ylabel("Přesnost [%]")
    axis.yaxis.label.set_fontsize("x-large")

    axis.yaxis.set_major_formatter(FuncFormatter(PERCENTAGE_FORMATTER))

    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    axis.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    axis.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    axis.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    plt.legend(loc="center left", bbox_to_anchor=(1.0125, 0.5), handles=DEFAULT_LEGEND)
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
        axis.set_title("Délka výpočtu deskriptoru části", fontsize="x-large")

    setupXAxis(axis)

    axis.set_ylabel("Délka výpočtu deskriptoru části [ms]")
    axis.yaxis.label.set_fontsize("x-large")
    axis.set_ylim(0, 30)

    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    axis.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    axis.bar(mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    axis.bar(largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    plt.legend(loc="center left", bbox_to_anchor=(1.0125, 0.5), handles=DEFAULT_LEGEND)
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
        bottom.set_title("Délka výpočtu deskriptoru obrázku", fontsize="x-large")

    setupXAxis(bottom)

    fig.text(0.07, 0.55, "Délka výpočtu deskriptoru obrázku [ms]", va="center", rotation="vertical", fontsize="x-large")

    bottom.set_ylim(0, 85)
    upper.set_ylim(100, 420)
    upper.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    bottom.bar(smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, upper], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    upper.legend(loc="center left", bbox_to_anchor=(1.0125, -0.1), handles=DEFAULT_LEGEND)
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

    fig, (upper, middle, bottom) = plt.subplots(3, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Průměrná velikost deskriptoru části", fontsize="x-large")

    fig.text(0.07, 0.55, "Průměrná velikost deskriptoru části", va="center", rotation="vertical", fontsize="x-large")

    setupXAxis(bottom)

    bottom.set_ylim(0, 8000)
    middle.set_ylim(10000, 30000)
    middle.xaxis.tick_top()
    upper.set_ylim(43000, 57000)
    upper.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    drawAcross([bottom, middle], smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, middle, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, middle, upper], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    middle.legend(loc="center left", bbox_to_anchor=(1.0125, 0.5), handles=DEFAULT_LEGEND)
    bottom.grid(True, axis="y")
    middle.grid(True, axis="y")
    upper.grid(True, axis="y")

    drawAxisSplitters(bottom, middle, upper)

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.15)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def imageDescriptorSize(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["descriptorImageSize"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["descriptorImageSize"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["descriptorImageSize"])

    fig, (upper, bottom) = plt.subplots(2, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Průměrná velikost deskriptoru obrázku", fontsize="x-large")

    setupXAxis(bottom)

    fig.text(0.05, 0.55, "Průměrná velikost deskriptoru obrázku", va="center", rotation="vertical", fontsize="x-large")

    bottom.set_ylim(0, 200000)
    upper.set_ylim(300000, 900000)
    upper.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    drawAcross([bottom, upper], smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, upper], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    upper.legend(loc="center left", bbox_to_anchor=(1.0125, -0.1), handles=DEFAULT_LEGEND)
    bottom.grid(True, axis="y")
    upper.grid(True, axis="y")

    drawAxisSplitters(bottom, upper)

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.15)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def matching(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["matchingSingle"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["matchingSingle"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["matchingSingle"])

    fig, (upper, middle, bottom) = plt.subplots(3, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Délka hledání v jednom obrázku", fontsize="x-large")

    setupXAxis(bottom)

    fig.text(0.07, 0.55, "Délka hledání v jednom obrázku [ms]", va="center", rotation="vertical", fontsize="x-large")

    bottom.set_ylim(0, 40)
    middle.set_ylim(80, 140)
    middle.xaxis.tick_top()
    upper.set_ylim(600, 700)
    upper.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    drawAcross([bottom, middle], smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, middle, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, middle], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    middle.legend(loc="center left", bbox_to_anchor=(1.0125, 0.5), handles=DEFAULT_LEGEND)
    bottom.grid(True, axis="y")
    middle.grid(True, axis="y")
    upper.grid(True, axis="y")

    drawAxisSplitters(bottom, middle, upper)

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.15)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def partProcess(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["partProcess"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["partProcess"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["partProcess"])

    fig, (upper, middle, bottom) = plt.subplots(3, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Délka zpracování celé části", fontsize="x-large")

    setupXAxis(bottom)

    fig.text(0.07, 0.55, "Délka zpracování celé části [ms]", va="center", rotation="vertical", fontsize="x-large")

    bottom.set_ylim(0, 600)
    middle.set_ylim(1000, 11000)
    middle.xaxis.tick_top()
    upper.set_ylim(46000, 52000)
    upper.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    drawAcross([bottom, middle], smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, middle, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, middle], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    middle.legend(loc="center left", bbox_to_anchor=(1.0125, 0.5), handles=DEFAULT_LEGEND)
    bottom.grid(True, axis="y")
    middle.grid(True, axis="y")
    upper.grid(True, axis="y")

    drawAxisSplitters(bottom, middle, upper)

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.15)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def totalTime(title=False, filename=None, show=False):
    smallX, smallY = splitIntoXY(data.DATA_300x300["totalTime"])
    mediumX, mediumY = splitIntoXY(data.DATA_640x480["totalTime"])
    largeX, largeY = splitIntoXY(data.DATA_1280x720["totalTime"])

    fig, (upper, middle, bottom) = plt.subplots(3, 1, sharex=True, figsize=PICTURE_SIZE)

    if title:
        bottom.set_title("Celkový čas", fontsize="x-large")

    fig.text(0.07, 0.55, "Celkový čas [s]", va="center", rotation="vertical", fontsize="x-large")

    setupXAxis(bottom)

    bottom.yaxis.set_major_formatter(FuncFormatter(SECOND_FORMATTER))
    middle.yaxis.set_major_formatter(FuncFormatter(SECOND_FORMATTER))
    upper.yaxis.set_major_formatter(FuncFormatter(SECOND_FORMATTER))

    bottom.set_ylim(0, 30000)
    middle.set_ylim(60000, 130000)
    middle.xaxis.tick_top()
    upper.set_ylim(250000, 2600000)
    upper.xaxis.tick_top()
    # up to 3 bars (sizes), 0.8 is default and leaves a little room between algorithms
    barWidth = 0.8 / 3
    hW = barWidth / 2

    drawAcross([bottom, middle, upper], smallX - 2 * hW, smallY, barWidth, color=pickColors(COLORS_300x300, smallX), edgecolor="black")
    drawAcross([bottom, middle, upper], mediumX, mediumY, barWidth, color=pickColors(COLORS_640x480, mediumX), edgecolor="black")
    drawAcross([bottom, middle, upper], largeX + 2 * hW, largeY, barWidth, color=pickColors(COLORS_1280x720, largeX), edgecolor="black")

    middle.legend(loc="center left", bbox_to_anchor=(1.0125, 0.5), handles=DEFAULT_LEGEND)
    bottom.grid(True, axis="y")
    middle.grid(True, axis="y")
    upper.grid(True, axis="y")

    drawAxisSplitters(bottom, middle, upper)

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(top=top, pictureSize=PICTURE_SIZE, **DEFAULT_MARGINS), hspace=0.15)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

def lighting(title=None, filename=None, show=False):
    def getData():
        x = []
        y = []
        z = []
        for b, val in data.DATA_LIGHTING_FT.items():
            for c, acc in val.items():
                x.append(b)
                y.append(c)
                z.append(acc)
        return np.asarray(x), np.asarray(y), np.asarray(z)

    colorMap = plt.cm.get_cmap("RdYlGn")
    x, y, z = getData()

    fig, axis = plt.subplots(figsize=PICTURE_SIZE)

    axis.set_ylabel("Kontrast", fontsize="x-large")
    axis.set_xlabel("Jas", fontsize="x-large")
    axis.set_ylim([-90, 90])
    axis.set_xlim([-90, 90])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f} %"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f} %"))

    for tick in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick.set_fontsize("large")

    sc = axis.scatter(x, y, c=z, vmin=0, vmax=1, s=40, cmap=colorMap)
    fig.colorbar(sc, format=FuncFormatter(PERCENTAGE_FORMATTER))
    for i in range(len(x)):
        _x = x[i]
        _y = y[i]
        acc = z[i]
        axis.annotate(PERCENTAGE_FORMATTER(acc, 0), (_x, _y), textcoords="offset pixels", xytext=(-10, 10))

    top = TOP_MARGIN_TITLE if title else TOP_MARGIN_NO_TITLE
    plt.subplots_adjust(**transformMargins(left=2.2, right=0.8, bottom=0.6, top=top, pictureSize=PICTURE_SIZE), hspace=0.15)

    if filename:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))
    if show:
        plt.show()

accuracy(filename="accuracy.png")
partDescriptorTime(filename="partDescriptorTime.png")
imageDescriptorTime(filename="imageDescriptorTime.png")
partDescriptorSize(filename="partDescriptorSize.png")
imageDescriptorSize(filename="imageDescriptorSize.png")
matching(filename="matching.png")
partProcess(filename="partProcess.png")
totalTime(filename="totalTime.png")
lighting(filename="lighting_ft.png")