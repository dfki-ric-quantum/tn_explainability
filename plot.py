# ruff: noqa: E402

import pickle

import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging
from matplotlib.colors import LinearSegmentedColormap

from tn.data.fold import TrainingKFold

COLORS = [
    "#1d3a8f",  # blue
    "#8b4c1f",  # brown
    "#ec619f",  # pink
    "#6abfa3",  # green
    "#f7a712",  # orange
    "#9b51e0",  # purple
]


def _load_ecg_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the ECG 5000 data set and samples for the plot

    Returns
    -------
    Returns the regular and anomaly samples for the plot as numpy arrays
    """
    CONT = 0.05
    SEED = 42
    IN_IDX = 0
    OUT_IDX = 295

    fold = TrainingKFold(
        dataset="ecg5000",
        contamination=CONT,
        n_splits=10,
        shuffle=True,
        scaler="minmax",
        seed=SEED,
        include_test=True,
    )

    _, x_test, _, _ = next(fold.split())
    return x_test[IN_IDX], x_test[OUT_IDX]


def _get_cmap() -> LinearSegmentedColormap:
    """Create a colormap for the Satellite plot.

    Returns
    -------
    A LinearSegmentedColormap object
    """
    colors = []
    for c in COLORS:
        _c = c.strip("#")
        colors.append(tuple(int(_c[i : i + 2], 16) / 255 for i in (0, 2, 4)))

    return LinearSegmentedColormap.from_list(
        "exptn", colors=[colors[0], colors[3], colors[4]]
    )


def plot_ecg(data: dict, outfile: str) -> None:
    """Create the ECG plot and save it to a PDF file.

    Parameters
    ----------
    data: dict
        The data loaded from the analysis
    outfile: str
        The pdf file to save the plot to

    """

    fig, ax = plt.subplots(
        2,
        1,
        sharex="col",
        figsize=(2.8, 3),
        constrained_layout=True,
    )

    inlier, outlier = _load_ecg_data()

    for a, dkey in zip(ax, ["mps", "ttn"]):
        mu = data[f"{dkey}_mu"]
        std = data[f"{dkey}_std"]
        steps = np.arange(0, mu.shape[0])

        a.plot(steps, mu, c=COLORS[3], lw=0.5, label="Expected")
        a.plot(steps, inlier, c=COLORS[0], lw=0.5, ls="dashed", label="Normal")
        a.plot(steps, outlier, c=COLORS[4], lw=0.5, ls="dashdot", label="Anomaly")
        a.fill_between(steps, mu + std, mu - std, color=COLORS[3], alpha=0.3, lw=0)
        a.set_xticks([])
        a.set_yticks([])
        a.set_ylim(0.0, 1.0)
        a.set_xlim(-5, 145)
        a.set_ylabel("Normalized Amplitude")

    bbox_props = dict(
        facecolor="none",
        edgecolor="black",
        lw=0.4,
        pad=2.0,
    )

    ax[0].text(133.8, 0.933, "MPS", color="black", bbox=bbox_props)
    ax[1].text(134, 0.933, "TTN", color="black", bbox=bbox_props)

    ax[1].set_xlabel("Normalized Time")
    ax[0].legend(
        ncols=5,
        fontsize=5,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.3,
        edgecolor="black",
        loc=(0.18, -0.17),
    ).get_frame().set_linewidth(0.2)

    fig.savefig(outfile, dpi=600)


def plot_sat(data: dict, outfile: str) -> None:
    """Create the Satellite plot and save it to a PDF file.

    Parameters
    ----------
    data: dict
        The data loaded from the analysis
    outfile: str
        The pdf file to save the plot to

    """
    fig, ax = plt.subplots(
        1,
        5,
        sharey="row",
        figsize=(5, 1.5),
        constrained_layout=True,
    )
    n_features = 36

    feature_ticks = np.arange(0, n_features, 10)
    titles = ["MPS", "TTN", "Data", "Normal", "Anomalies"]

    mps_mi = data["mps"]["mi"]
    ttn_mi = data["ttn"]["mi"]
    data_mi = data["all"]["mi"]
    inlier_mi = data["inlier"]["mi"]
    outlier_mi = data["outlier"]["mi"]

    for mi in [mps_mi, ttn_mi, data_mi, inlier_mi, outlier_mi]:
        mask = np.ones(mi.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        min_value = mi[mask].min()

        mi -= min_value
        max_value = mi[mask].max()
        mi /= max_value
        np.fill_diagonal(mi, 0.0)

    vmin = 0.0
    vmax = np.max(
        [data_mi.max(), inlier_mi.max(), outlier_mi.max(), mps_mi.max(), ttn_mi.max()]
    )

    for a, corr, t in zip(
        ax.ravel(), [mps_mi, ttn_mi, data_mi, inlier_mi, outlier_mi], titles
    ):
        a.set_title(t)
        im = a.imshow(corr, cmap=_get_cmap(), vmin=vmin, vmax=vmax)
        a.set_xticks(feature_ticks)
        a.set_yticks(feature_ticks)
        a.set_xlabel("Feature")
        a.tick_params(axis="both", length=2, width=0.4, pad=1.5)

    ax[0].set_ylabel("Feature")
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.008)

    fig.savefig(outfile, dpi=600)


def plot_spam(data: dict, outfile: str) -> None:
    """Create the Spambase plot and save it to a PDF file.

    Parameters
    ----------
    data: dict
        The data loaded from the analysis
    outfile: str
        The pdf file to save the plot to

    """
    fig, ax = plt.subplots(1, 1, figsize=(4.33, 1.5), constrained_layout=True)
    n_features = 57

    features = np.arange(1, n_features + 1)
    feature_ticks = np.arange(1, n_features + 1, 5)

    y_lower = np.clip(data["mps_std"], 0.0, data["mps_mu"])

    ax.errorbar(
        features,
        data["mps_mu"],
        (y_lower, data["mps_std"]),
        fmt="o",
        markersize=2,
        lw=0.5,
        color=COLORS[0],
        alpha=0.75,
        label="MPS Dist.",
    )
    ax.bar(
        features - 0.25,
        data["outlier"],
        width=0.5,
        color=COLORS[3],
        alpha=0.75,
        label="Outlier",
    )
    ax.bar(
        features + 0.25,
        data["corrected"],
        width=0.5,
        color=COLORS[4],
        alpha=0.75,
        label="Corrected",
    )
    ax.set_xticks(feature_ticks)
    ax.tick_params(axis="both", length=2, width=0.4, pad=1.5)
    ax.set_xlim(0.0, 58.0)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Rel. Feature Freq.")
    ax.legend(
        fontsize=5,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.3,
        edgecolor="black",
        loc="upper left",
    ).get_frame().set_linewidth(0.2)

    fig.savefig(outfile, dpi=600)


FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "plot", None, ["ecg", "sat", "spam"], short_name="p", help="Plot type"
)
flags.DEFINE_string("infile", None, short_name="i", help="File to read input from")
flags.DEFINE_string(
    "outfile", default=None, short_name="o", help="The PDF file to write the plot to"
)


def main(_):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "axes.linewidth": 0.4,
            "font.family": "serif",
            "font.serif": "Times",
            "font.size": 6,
        }
    )

    with open(FLAGS.infile, "rb") as infile:
        data = pickle.load(infile)

    match FLAGS.plot:
        case "ecg":
            plot_ecg(data, FLAGS.outfile)
        case "sat":
            plot_sat(data, FLAGS.outfile)
        case "spam":
            plot_spam(data, FLAGS.outfile)
        case _:
            logging.error("No such plot: %s", FLAGS.plot)


if __name__ == "__main__":
    app.run(main)
