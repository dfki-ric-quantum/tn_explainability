# Explaining Anomalies with Tensor Networks

This repository contains the implementation of all methods presented in [1], including data
preparation, hyper-parameter tuning, training and evaluation for three baseline models
(one-class SVM, Isolation Forest, Autoencoder) as well as two tensor network models (matrix product
state and tree tensor network).

## Requirements

To install the requirements for this repository run either of the three commands below:
```bash
pip install -r requirements_nocuda.txt       # if you don't want/need CUDA
pip install -r requirements_withcuda.txt     # if you want to install jax together with CUDA
pip install -r requirements_localcuda.txt    # if you have a local CUDA installation
```

All requirements beyond `jax` itself are maintained in `base_requirements.txt`.

## Preparation

Before the models can be trained to reproduce results as discussed in [1], several preparations have
to be performed first.

### Download the ECG 5000 data set

The ECG 5000 data set is not part of this repository. To download the data set and prepare it in the
correct format used by the implementation, run the following script:

```bash
./download_ecg.sh
```

**Note:** This will only work on Linux and UNIX like systems.

The other two data sets are provided with the [ucimlrep](https://github.com/uci-ml-repo/ucimlrepo)
package. Since the package attempts to download the requested data set on every call, we provide a
convenience wrapper, that caches the downloaded file to `~/.cache/dfki_exptn/`, which only works on
Linux. To cache to another path, edit the function `_linux_path()` in `tn/data/fetch.py`.


### Generate Anomalies

In the experiments discussed in [1], we used anomalies/outliers provided with the respective data
sets and generated anomalies of different types as described in [2]. To generate a fixed set of
anomalies, run the provided script:

```bash
python3 gen_anomalies.py -d all
```

This will generate all required artificial anomaly samples for all three data sets. Alternatively
the script can be run for a single data set e.g., via

```bash
python3 gen_anomalies.py -d ecg5000
```

To only generate anomalies for the ECG 5000 data set. See `--help` for details. The generated
anomalies will be saved as `*.pickle` files in `data/datasets/anomalies`. Generating all anomalies
will take about 10 minutes on a AMD Ryzen 3700X system.


### Hyper-parameter tuning (Optional)

We provide the setup we used in [1] to tune all models on each data set for completeness. The
original tuning was performed in several steps, narrowing down the search spaces iteratively and
fine tuning individual parameters. Since this process is tedious and depending on available hardware
setup can take days, if not weeks, all hyper-parameter configurations we used are stored in
`config/`. All hyper-parameter tuning is done with [optuna](https://optuna.org/), tuning results are
stored in a local sqlite database `tune.db` which can be evaluated e.g., with `optuna-dashboard`
before storing the results in the format outlined in `config/`

#### Tuning SVM and Isolation forest

The two simple models can be tuned with `tune_simple.py`:
```
       USAGE: tune_simple.py [flags]
flags:

tune_simple.py:
  -d,--dataset: <satellite|ecg5000|spambase>: Dataset to tune on
  -e,--estimator: <svm|if>: Estimator to tune
  -t,--trials: Number of trials
    (default: '100')
    (an integer)
```
To adapt the search space, you have to edit `tn/tune/simple.py`.

#### Tuning the Autoencoder

To tune the autoencoder, run `tune_ae.py`:

```
       USAGE: tune_ae.py [flags]
flags:

tune_ae.py:
  -d,--dataset: <satellite|ecg5000|spambase>: Dataset to tune on
  -t,--trials: Number of trials
    (default: '100')
    (an integer)
```
To adapt the search space, you have to edit `tn/tune/ae.py`.

#### Tuning Tensor Networks

To tune the tensor networks, run `tune_tn.py`:

```
       USAGE: tune_tn.py [flags]
flags:

tune_tn.py:
  -d,--dataset: <satellite|ecg5000|spambase>: Dataset to tune on
  --tn: <mps|ttn>: Tensor network
  -t,--trials: Number of trials
    (default: '30')
    (an integer)
```
To adapt the search space, you have to edit `tn/tune/mps.py` and `tn/tune/ttn.py`.


## Training

### Training Baseline Models

To train SVM, Isolation Forest or autoencoder, run `train_baseline.py`:
```
       USAGE: train_baseline.py [flags]
flags:

train_baseline.py:
  --config: path to config file.
    (default: 'None')
  -c,--contamination: Outlier contamination
    (default: '0.05')
    (a number)
  -d,--dataset: <satellite|ecg5000|spambase>: Dataset
  -f,--folds: Number fo training folds
    (default: '10')
    (an integer)
  -m,--model: <svm|ifo|ae>: model type
    (default: 'svm')
  -o,--outdir: Output directory
    (default: './')
  -s,--seed: Random seed
    (an integer)
```

The `--config` flag loads the correct hyper-parameter setup. E.g. to train an SVM on the Satellite
data set for 10 folds and save the trained model to `results/`, run:
```
python3 train_baseline.py --config=config/svm.py:satellite -d satellite -f 10 -m svm -o results/ --seed 42
```
Each of the 10 trained models will be saved in a separate `*.pickle` file. The only exception is the
autoencoder, which uses [Orbax](https://github.com/google/orbax) checkpoints to save the model.

The training for all three baseline models is rather quick and should be done within a few minutes.

### Training Tensor Networks

To train either of the two tensor network models (MPS or TTN), run `train_tn.py`:
```
       USAGE: train_tn.py [flags]
flags:

train_tn.py:
  --config: path to config file.
    (default: 'None')
  -c,--contamination: Outlier contamination
    (default: '0.05')
    (a number)
  -d,--dataset: <satellite|ecg5000|spambase>: Dataset
  -f,--folds: Number fo training folds
    (default: '10')
    (an integer)
  -m,--model: <mps|ttn>: model type
    (default: 'mps')
  -o,--outdir: Output directory
    (default: './')
  -s,--seed: Random seed
    (an integer)
```
The training script works similar to the baseline model training, e.g. to train a matrix product
state for 10 folds on the ECG 5000 data set, run:
```
python3 train_tn.py --config=config/mps.py:ecg5000 -d ecg5000 -f 10 -m mps -o results/ --seed 42
```

**Note:** Training of the tensor networks may take several hours, depending on available CPU and
GPU and the chosen data set more than a day.


## Evaluation

The evaluation process to produce Table II and Figures 3a-3c in [1] is organized in two steps. First
the necessary data is collected and in a second step, the desired output is produced.

### Data Collection

Data collection is performed via `analyze.py`:
```
       USAGE: analyze.py [flags]
flags:

analyze.py:
  --config: path to config file.
    (default: 'None')
  -o,--outfile: Output file
```
The configuration for each analysis is stored in `config/analyze.py`, which you may need to edit, to
adjust file paths to trained models. There are four different types of data collections that can be
performed:

* `roc`: Collects statistics on ROC scores for all model types and data sets, used in Table II of [1].
* `ecg`: Computes the input data for model/outlier comparison in Fig. 3a of [1].
* `sat`: Collects mutual-information and all-to-all feature correlations for the Satellite data set,
    used in Fig. 3b of [1].
* `spam`: Computes feature corrections based on a matrix product state for the Spambase data set,
    used in Fig. 3c of [1].

To run e.g. the `roc` analysis, run:
```
python3 analyze.py --config=config/analyze.py:roc -o roc.pickle
```
This will save the collected data to `roc.pickle`, which can be consumed in the next step.

### Plots / Tables

The script `table.py` can be used to create a table on the ROC statistics similar to Table II in
[1]:
```
      USAGE: table.py [flags]
flags:

table.py:
  -f,--format: <latex|html|csv|markdown>: Output format
    (default: 'latex')
  -i,--infile: The file to load
  -t,--task: <sep|ind>: Task, sep for separation, ind for inductive
    (default: 'sep')
```
The input file is the one created with the `roc` analysis in the previous step. The two tasks are
the ones described in [1]. The output will be printed to the command line.

To create the plots in Fig. 3a-3c in [1], use `plot.py`:
```
       USAGE: plot.py [flags]
flags:

plot.py:
  -i,--infile: File to read input from
  -o,--outfile: The PDF file to write the plot to
  -p,--plot: <ecg|sat|spam>: Plot type
```
The script receives the file generated by `analyze.py` in the previous step. E.g., to create the ECG
5000 plot, assuming you saved the analysis results to `ecg5000_analysis.pickle`, run
```
python3 plot.py -i ecg5000_analysis.pickle -p ecg -o ecg5000.pdf
```
This will save the plot to `ecg5000.pdf`.

**Note**: The plots will not look exactly as in [1], as certain details such as emphasising/marking
relevant features and spacing have been adjusted manually.


## Unit tests

There are a some unit tests in `test/`. To run them, execute:
```
python3 -m unittest
```
from the top-level directory. This will skip tests marked as rather slow (mostly those involving
heavy use of numerical integration) and should run within a few seconds.
To run all, including the slow ones:
```
TN_SLOW_TEST=1 python3 -m unittest
```
This will take around 10 minutes to complete.


## Acknowledgment

This work was funded by the German Ministry of Economic Affairs and Climate Action (BMWK) and the
German Aerospace Center (DLR) in project QuDA-KI (grantno. 50RA2206).


## References

[1] Hohenfeld, H., Beuerle, M., & Mounzer, E. (2025). _Explaining Anomalies with Tensor Networks_. arXiv preprint arXiv:2505.03911.

[2] Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022). _Adbench: Anomaly detection benchmark_. Advances in neural information processing systems, 35, 32142-32159.
