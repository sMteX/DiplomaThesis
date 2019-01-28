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

def getVariableValues(variableName, dataset = data.DATA_SINGLE):
    return list(dataset[variableName].values())

def getVariableKeys(variableName, dataset = data.DATA_SINGLE):
    return list(dataset[variableName].keys())

def plotAndSave(dataset, variableName, filename=None, show=False):
    if CHART_DATA[variableName]["split"]:
        plotAndSaveTwinAxes(dataset, variableName, filename, show)
    else:
        plotAndSaveSingleAxis(dataset, variableName, filename, show)

def plotAndSaveSingleAxis(dataset, variableName, filename=None, show=False):
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


def plotAndSaveTwinAxes(dataset, variableName, filename=None, show=False):
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


print("Generating charts...")

for var in VARIABLES:
    plotAndSave(data.DATA_SINGLE, var, f"single/{var}.png")

print("Done!")