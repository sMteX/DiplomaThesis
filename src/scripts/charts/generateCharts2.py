import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import src.scripts.charts.data as data

OUTPUT_DIR = "./output"

COLORS = ["#cc00cc", "#e00000", "#e07000", "#efcf00", "#009e02", "#006ce0", "#7300e0"]

def plotSingleAxisSingleData(data, yAxisLabel,
                             title=None, percentage=False, filename=None, show=False):
    algorithms = list(data.keys())
    values = list(data.values())
    fig, ax = plt.subplots()

    index = np.arange(len(algorithms))

    if not title is None:
        ax.set_title(title)

    ax.set_xlabel("Algoritmus")
    ax.xaxis.label.set_fontsize("x-large")
    ax.set_xticks(index)
    ax.set_xticklabels(algorithms)
    for tick in ax.get_xticklabels():
        tick.set_fontsize("large")

    ax.set_ylabel(yAxisLabel)
    if len(yAxisLabel) <= 30:
        ax.yaxis.label.set_fontsize("x-large")
    else:
        ax.yaxis.label.set_fontsize("large")

    if percentage:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{(100 * y):.0f} %"))

    ax.bar(index, values, color=COLORS, edgecolor="black")

    plt.grid(True, axis="y")
    plt.tight_layout()

    if show:
        plt.show()
    if not filename is None:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))


def plotTwinAxesSingleData(leftData, rightData, leftYAxisLabel, rightYAxisLabel,
                           title=None, filename=None, show=False):
    leftAlgorithms = list(leftData.keys())
    leftValues = list(leftData.values())
    rightAlgorithms = list(rightData.keys())
    rightValues = list(rightData.values())
    fig, left = plt.subplots()

    algorithms = leftAlgorithms + rightAlgorithms
    index = np.arange(len(algorithms))

    if not title is None:
        left.set_title(title)

    left.set_xlabel("Algoritmus")
    left.xaxis.label.set_fontsize("x-large")
    left.set_xticks(index)
    left.set_xticklabels(algorithms)
    for tick in left.get_xticklabels():
        tick.set_fontsize("large")

    left.set_ylabel(leftYAxisLabel)
    if len(leftYAxisLabel) <= 30:
        left.yaxis.label.set_fontsize("x-large")
    else:
        left.yaxis.label.set_fontsize("large")

    # assume HOG is on the first index
    left.bar(index[:len(leftAlgorithms)], leftValues, color=COLORS[:len(leftAlgorithms)], edgecolor="black")

    right = left.twinx()

    right.set_ylabel(rightYAxisLabel)
    if len(rightYAxisLabel) <= 30:
        right.yaxis.label.set_fontsize("x-large")
    else:
        right.yaxis.label.set_fontsize("large")

    right.bar(index[len(leftAlgorithms):], rightValues, color=COLORS[len(leftAlgorithms):], edgecolor="black") # except first (which is HOG)

    plt.grid(True, axis="y")
    plt.tight_layout()

    if show:
        plt.show()
    if not filename is None:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

# assume it's pairs of data, all containing same number of items
def plotTwinAxesTwinData(leftData, rightData, leftYAxisLabel, rightYAxisLabel, leftLegend, rightLegend,
                         title=None, filename=None, show=False):
    fig, left = plt.subplots()

    right = left.twinx()

    labels = list(leftData.keys())
    leftValues = list(leftData.values())
    rightValues = list(rightData.values())

    barWidth = 0.35

    index = np.arange(len(labels))

    if not title is None:
        left.set_title(title)

    left.set_xlabel("Algoritmus")
    left.xaxis.label.set_fontsize("x-large")
    left.set_xticks(index)
    left.set_xticklabels(labels)
    for tick in left.get_xticklabels():
        tick.set_fontsize("large")

    left.set_ylabel(leftYAxisLabel)
    if len(leftYAxisLabel) <= 30:
        left.yaxis.label.set_fontsize("x-large")
    else:
        left.yaxis.label.set_fontsize("large")
    right.set_ylabel(rightYAxisLabel)
    if len(rightYAxisLabel) <= 30:
        right.yaxis.label.set_fontsize("x-large")
    else:
        right.yaxis.label.set_fontsize("large")


    l1 = left.bar(index - barWidth / 2, leftValues, barWidth, color=COLORS, edgecolor="black")
    l2 = right.bar(index + barWidth / 2, rightValues, barWidth, color=COLORS, hatch="x", edgecolor="black")

    plt.legend([l1, l2], (leftLegend, rightLegend))
    plt.grid(True, axis="y")

    plt.tight_layout()
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

print("Generating charts...")

plotSingleAxisSingleData(data=data.DATA_SINGLE["accuracy"],
                         yAxisLabel="Přesnost [%]",
                         percentage=True,
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
                         percentage=True,
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
                       leftYAxisLabel="Celkový čas (HOG) [ms]",
                       rightYAxisLabel="Celkový čas [ms]",
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

print("Done!")