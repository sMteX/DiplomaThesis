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
        "type": "number"
    },
    "descriptorPart": {
        "title": "Spočtení deskriptoru pro hledanou část",
        "yAxisLabel": Y_AXIS_TIME,
        "type": "number"
    },
    "descriptorImage": {
        "title": "Spočtení deskriptoru pro jednotlivý obrázek",
        "yAxisLabel": Y_AXIS_TIME,
        "type": "number"
    },
    "matchingSingle": {
        "title": "Hledání části v jednotlivém obrázku",
        "yAxisLabel": Y_AXIS_TIME,
        "type": "number"
    },
    "matchingAll": {
        "title": "Hledání části ve všech obrázcích",
        "yAxisLabel": Y_AXIS_TIME,
        "type": "number"
    },
    "partProcess": {
        "title": "Zpracování celé části",
        "yAxisLabel": Y_AXIS_TIME,
        "type": "number"
    },
    "descriptorPartSize": {
        "title": "Průměrná velikost deskriptoru části",
        "yAxisLabel": Y_AXIS_SIZE,
        "type": "number"
    },
    "descriptorImageSize": {
        "title": "Průměrná velikost deskriptoru obrázku",
        "yAxisLabel": Y_AXIS_SIZE,
        "type": "number"
    },
    "accuracy": {
        "title": "Přesnost hledání",
        "yAxisLabel": Y_AXIS_ACCURACY,
        "type": "percentage"
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
COLORS = ["#cb00d1", "#f51f44", "#ff9500", "#ffe40c", "#00ff72", "#00d8ff", "#0055ff"]

def getVariableValues(variableName, dataset = data.DATA_SINGLE):
    return list(dataset[variableName].values())

def getVariableKeys(variableName, dataset = data.DATA_SINGLE):
    return list(dataset[variableName].keys())

def plotAndSave(dataset, variableName, filename=None, show=False):
    algorithms = getVariableKeys(variableName, dataset)
    values = getVariableValues(variableName, dataset)
    fig, ax = plt.subplots()

    index = np.arange(len(algorithms))

    ax.bar(index, values, color=COLORS)
    ax.set_xlabel("Algoritmus")
    ax.set_ylabel(CHART_DATA[variableName]["yAxisLabel"])
    if CHART_DATA[variableName]["type"] == "percentage":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{(100 * y):.0f} %"))
    ax.set_title(CHART_DATA[variableName]["title"])
    ax.set_xticks(index)
    ax.set_xticklabels(algorithms)
    plt.grid(True, axis="y", zorder=0)

    if show:
        plt.show()
    if not filename is None:
        plt.savefig(os.path.abspath(f"{OUTPUT_DIR}/{filename}"))

print("Generating charts...")

for var in VARIABLES:
    plotAndSave(data.DATA_SINGLE, var, f"single/{var}.png")

print("Done!")