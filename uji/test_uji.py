"""
Test UJI

KNOWN ISSUES:
* Location labels over months correspond to different coordinates
"""
from uji import UJI
import logging as lg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


def test():
    trn_1 = UJI("train", 1)
    lg.debug(trn_1.__dict__)


def plot_test_months():
    fig, axs = plt.subplots(nrows=5, ncols=3)

    zoom = 3
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.set_figwidth(zoom * w)
    fig.set_figheight(zoom * 3 * h)

    for i, ax in tqdm(enumerate(axs.flatten()), total=15):
        #print("Month", i + 1)

        records = UJI.from_cache("test", i + 1).records

        records[["DATE"]] = records.TIMESTAMP.dt.date

        df = records[["DATE", "COLLECTION_INSTANCE",
                      "LABEL", "CORD_X", "CORD_Y"]].drop_duplicates()

        sns.scatterplot(
            data=df,
            x="CORD_X",
            y="CORD_Y",
            hue="COLLECTION_INSTANCE",
            style="DATE",
            palette="tab10",
            # s=100,
            ax=ax
        )

        label_np = df[["CORD_X", "CORD_Y", "LABEL"]].drop_duplicates().values
        for [x, y, lbl] in label_np:
            ax.annotate(str(int(lbl)), (x, y))

        im = plt.imread("floorplan.png")
        implot = ax.imshow(im, extent=[2.5, 14.5, 14.65, 31.3])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Test Data for {i + 1}th Month")

    plt.tight_layout()
    plt.savefig("test_uji.png", dpi=150)
    # plt.show()


if __name__ == "__main__":
    # lg.basicConfig(level=lg.DEBUG)
    # test()
    print("Building data collection plots for test months\n\n\n")
    lg.basicConfig(level=lg.CRITICAL)
    plot_test_months()
