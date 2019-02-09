import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import src.scripts.charts.data as data

OUTPUT_DIR = "./output"
Y_AXIS_TIME = "Čas [ms]"
Y_AXIS_SIZE = "Velikost deskriptoru"
Y_AXIS_ACCURACY = "Přesnost [%]"

CHART_DATA = {
    "totalTime": {
        "title": "Celkový čas",
        "yAxisLabel": Y_AXIS_TIME,
        "percentage": False,
        "split": True,
    },
    "descriptorPart": {
        "title": "Výpočet deskriptoru pro hledanou část",
        "yAxisLabel": Y_AXIS_TIME,
        "percentage": False,
        "split": False,
    },
    "descriptorImage": {
        "title": "Výpočet deskriptoru pro jednotlivý obrázek",
        "yAxisLabel": Y_AXIS_TIME,
        "percentage": False,
        "split": False,
    },
    "matchingSingle": {
        "title": "Hledání části v jednotlivém obrázku",
        "yAxisLabel": Y_AXIS_TIME,
        "percentage": False,
        "split": True,
    },
    "matchingAll": {
        "title": "Hledání části ve všech obrázcích",
        "yAxisLabel": Y_AXIS_TIME,
        "percentage": False,
        "split": True,
    },
    "partProcess": {
        "title": "Zpracování celé části",
        "yAxisLabel": Y_AXIS_TIME,
        "percentage": False,
        "split": True,
    },
    "descriptorPartSize": {
        "title": "Průměrná velikost deskriptoru části",
        "yAxisLabel": Y_AXIS_SIZE,
        "percentage": False,
        "split": False,
    },
    "descriptorImageSize": {
        "title": "Průměrná velikost deskriptoru obrázku",
        "yAxisLabel": Y_AXIS_SIZE,
        "percentage": False,
        "split": False,
    },
    "accuracy": {
        "title": "Přesnost hledání",
        "yAxisLabel": Y_AXIS_ACCURACY,
        "percentage": True,
        "split": False,
    },
}
VARIABLES = [
    "totalTime",
    "descriptorPart",
    "descriptorImage",
    "matchingSingle",
    "matchingAll",
    "partProcess",
    "descriptorPartSize",
    "descriptorImageSize",
    "accuracy"
]
COLORS = ["#cc00cc", "#e00000", "#e07000", "#efcf00", "#009e02", "#006ce0", "#7300e0"]

def getVariableValues(variableName, dataset = data.DATA_300x300_SMALL):
    return list(dataset[variableName].values())

def getVariableKeys(variableName, dataset = data.DATA_300x300_SMALL):
    return list(dataset[variableName].keys())

def plotAndSave(dataset, variableName, filename=None, show=False):
    if CHART_DATA[variableName]["split"]:
        plotTwinAxesSingleData(dataset, variableName, filename, show)
    else:
        plotSingleAxisSingleData(dataset, variableName, filename, show)

def plotSingleAxisSingleData(dataset, variableName, filename=None, show=False):
    algorithms = getVariableKeys(variableName, dataset)
    values = getVariableValues(variableName, dataset)
    fig, ax = plt.subplots()

    index = np.arange(len(algorithms))

    ax.set_title(CHART_DATA[variableName]["title"])

    ax.set_xlabel("Algoritmus")
    ax.set_xticks(index)
    ax.set_xticklabels(algorithms)

    ax.set_ylabel(CHART_DATA[variableName]["yAxisLabel"])
    if CHART_DATA[variableName]["percentage"]:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{(100 * y):.0f} %"))

    ax.bar(index, values, color=COLORS)

    plt.grid(True, axis="y")

    if show:
        plt.show()
    if not filename is None:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))


def plotTwinAxesSingleData(dataset, variableName, filename=None, show=False):
    algorithms = getVariableKeys(variableName, dataset)
    values = getVariableValues(variableName, dataset)
    fig, ax = plt.subplots()

    index = np.arange(len(algorithms))

    ax.set_title(CHART_DATA[variableName]["title"])

    ax.set_xticks(index)
    ax.set_xticklabels(algorithms)
    ax.set_xlabel("Algoritmus")

    ax.set_ylabel(f"{CHART_DATA[variableName]['yAxisLabel']} (HOG)")

    # assume HOG is on the first index
    ax.bar(index[:1], values[:1], color=COLORS[:1])

    ax2 = ax.twinx()

    ax2.set_ylabel(CHART_DATA[variableName]["yAxisLabel"])

    ax2.bar(index[1:], values[1:], color=COLORS[1:]) # except first (which is HOG)

    plt.grid(True, axis="y")

    if show:
        plt.show()
    if not filename is None:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

def testQuadrupleYAxis():
    COLORS2 = ["r", "g", "b", "c", "m", "y", "k"]
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig, left1 = plt.subplots()

    right1 = left1.twinx()
    left2 = left1.twinx()
    right2 = left1.twinx()

    make_patch_spines_invisible(right1)
    make_patch_spines_invisible(right2)
    make_patch_spines_invisible(left2)

    right2.spines["right"].set_position(("axes", 1.2))
    left2.spines["left"].set_position(("axes", -0.2))

    right2.spines["right"].set_visible(True)
    left2.spines["left"].set_visible(True)
    left2.yaxis.set_label_position("left")
    left2.yaxis.set_ticks_position("left")

    labels = getVariableKeys("matchingSingle")
    matchingAll = getVariableValues("matchingAll", data.DATA_300x300_SMALL)
    matchingSingle = getVariableValues("matchingSingle", data.DATA_300x300_SMALL)

    barWidth = 0.35

    index = np.arange(len(labels))

    left1.set_title("Hledání části")
    left1.set_xlabel("Algoritmus")
    left1.set_xticks(index)
    left1.set_xticklabels(labels)

    left1.set_ylabel("v jednotlivém obrázku (HOG)")
    left2.set_ylabel("ve všech obrázcích (HOG)")
    right1.set_ylabel("v jednotlivém obrázku")
    right2.set_ylabel("ve všech obrázcích")

    left2.bar(index[:1] - barWidth/2, matchingAll[:1], barWidth, color=COLORS2[:1])
    left1.bar(index[:1] + barWidth/2, matchingSingle[:1], barWidth, color=COLORS[:1])
    right2.bar(index[1:] - barWidth / 2, matchingAll[1:], barWidth, color=COLORS2[1:])
    right1.bar(index[1:] + barWidth / 2, matchingSingle[1:], barWidth, color=COLORS[1:])

    tkw = dict(size=4, width=1.5)
    left1.tick_params(axis='y', **tkw)
    right1.tick_params(axis='y', **tkw)
    right2.tick_params(axis='y', **tkw)
    left2.tick_params(axis='y', **tkw)
    left1.tick_params(axis='x', **tkw)

    plt.tight_layout()
    plt.grid(True, axis="y")
    plt.show()

def testTwinAxesTwinData():
    fig, left = plt.subplots()

    right = left.twinx()

    labels = getVariableKeys("descriptorImage")
    descriptorPart = getVariableValues("descriptorPart", data.DATA_300x300_SMALL)
    descriptorImage = getVariableValues("descriptorImage", data.DATA_300x300_SMALL)

    barWidth = 0.35

    index = np.arange(len(labels))

    left.set_title("Výpočet deskriptoru")
    left.set_xlabel("Algoritmus")
    left.set_xticks(index)
    left.set_xticklabels(labels)

    left.set_ylabel("části - čas [ms]")
    right.set_ylabel("obrázku - čas [ms]")

    l1 = left.bar(index - barWidth / 2, descriptorPart, barWidth, color=COLORS, edgecolor="black")
    l2 = right.bar(index + barWidth / 2, descriptorImage, barWidth, color=COLORS, hatch="x", edgecolor="black")

    plt.legend([l1, l2], ("Výpočet deskriptoru části", "Výpočet deskriptoru obrázku"))
    plt.grid(True, axis="y")
    plt.show()

# print("Generating charts...")
#
# for var in VARIABLES:
#     plotAndSave(data.DATA_300x300_SMALL, var, f"single/{var}.png")
#
# print("Done!")