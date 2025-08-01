import os
import pickle

from absl import app, flags, logging

from tn.data.anomalies import generate_all_anomalies
from tn.data.loader import load_dataset

DATASETS = [
    "ecg5000",
    "satellite",
    "spambase",
]

CONT = 0.1
SEED = 42
N_ANOMALY_TYPES = 4


FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "dataset", None, enum_values=DATASETS + ["all"], short_name="d", help="dataset"
)


def main(_):
    dataset = DATASETS if FLAGS.dataset == "all" else [FLAGS.dataset]

    for ds in dataset:
        logging.info("Generating anomalies for %s", ds)

        inliers, outliers = load_dataset(ds)

        n_inliers = inliers.shape[0]
        n_outliers = outliers.shape[0]

        n_total_outliers = int(CONT * n_inliers / (1 - CONT))
        n_natural_outliers = min(n_total_outliers // 2, n_outliers)
        n_gen_outliers = n_total_outliers - n_natural_outliers

        anomalies = generate_all_anomalies(
            inliers, n_anomalies=n_gen_outliers // N_ANOMALY_TYPES, seed=SEED
        )

        os.makedirs("data/datasets/anomalies/", exist_ok=True)
        fname = f"data/datasets/anomalies/{ds}_gen.pickle"

        with open(fname, "wb") as outfile:
            pickle.dump(anomalies, outfile)

        logging.info("Saved %d generated anomalies to %s", n_gen_outliers, fname)


if __name__ == "__main__":
    app.run(main)
