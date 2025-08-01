import pickle
from itertools import product

import pandas as pd
from absl import app, flags, logging

DATASETS = ["ecg5000", "satellite", "spambase"]
SKEYS = ["mean", "std"]

FLAGS = flags.FLAGS
flags.DEFINE_string("infile", default=None, short_name="i", help="The file to load")
flags.DEFINE_enum(
    "task",
    default="sep",
    enum_values=["sep", "ind"],
    short_name="t",
    help="Task, sep for separation, ind for inductive",
)
flags.DEFINE_enum(
    "format",
    default="latex",
    enum_values=["latex", "html", "csv", "markdown"],
    short_name="f",
    help="Output format",
)


def get_stat(data: dict, task: str, dataset: str, skey: str) -> dict:
    """Get statistic

    Parameters
    ----------
    data: dict
        The ROC statistics as read from the analyis file
    task: str
        The task to read the statistic for, either 'sep' or 'ind'
    dataset: str
        The dataset to read the statistic for
    skey: str
        Statistics key, either 'mean' or 'std'

    Returns
    -------
    A dictonary with each model type as key and the requested statistic as
    value
    """
    res = {}

    sidx = 0 if skey == "mean" else 1

    for model, stats in data[dataset].items():
        match task:
            case "sep":
                res[model] = stats.train[sidx]
            case "ind":
                res[model] = stats.test[sidx]
            case _:
                logging.error("Unkown task: %s", task)

    return res


def main(_) -> None:
    with open(FLAGS.infile, "rb") as infile:
        data = pickle.load(infile)

        df = pd.DataFrame(
            {col: get_stat(data, FLAGS.task, *col) for col in product(DATASETS, SKEYS)}
        )

        match FLAGS.format:
            case "latex":
                print(df.to_latex())
            case "html":
                print(df.to_html())
            case "csv":
                print(df.to_csv())
            case "markdown":
                print(df.to_markdown())
            case _:
                logging.error("Unsupported format: %s", FLAGS.format)


if __name__ == "__main__":
    app.run(main)
