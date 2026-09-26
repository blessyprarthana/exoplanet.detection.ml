# Automated Exoplanet Detection from Space Telescope Data

COMP702 MSc Project - Prarthana Voosala (201946897)

This project looks at whether machine learning can be used to classify NASA Kepler Objects of Interest (KOIs) as either possible planets or false positives.

The project includes the main stages of the process:

- cleaning the KOI catalogue;
- removing columns that would give the model information about the final answer;
- training and comparing Logistic Regression, Random Forest and MLP models;
- tuning the models;
- testing them on data from stars that were not used during training; and
- running the trained models through a Streamlit app.

## The data

The data comes from the cumulative KOI table on the NASA Exoplanet Archive. I downloaded it on 19 August 2026 from:

[https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)

The archive is updated regularly, so downloading the table again may produce slightly different results. The version used in this project is stored in `data/datacumulative_koi.csv`.

After filtering the table to the three relevant dispositions, there are 9,564 KOIs.

For this project, I have grouped CONFIRMED and CANDIDATE objects into the positive class, with FALSE POSITIVE as the negative class. This means the model is answering the question:

> Is this signal worth investigating further?

It is not intended to decide whether an object has definitely been confirmed as a planet.

## Setup

The project requires Python 3.11 or newer. From the project folder, install the dependencies with:

```bash
pip install -r requirements.txt
```

scikit-learn is pinned to version 1.9.0 because the saved pipelines in `models/` were trained with it. Loading them under a different minor version raises a warning and can change the predictions.

## Running it

The notebook produces everything else:

```bash
jupyter notebook exoplanet_detection.ipynb
```

Run it from top to bottom. It takes about five minutes on a laptop, most of which is the MLP and the hyperparameter search, and it writes the fitted pipelines into `models/` and the metrics and figures into `results/`.

The Streamlit demo reads what the notebook produced:

```bash
streamlit run app.py
```

Upload `test_dataset_20_percent.csv` to try it on the held-out test set, or `test_false_positive_example.csv` for a single row the models classify as a false positive. Both files are written by the notebook.

## Results

Scores on the held-out 20% test set, which is split by host star so that no star appears in both training and test:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.829 | 0.884 | 0.855 | 0.938 |
| Random Forest | 0.914 | 0.876 | **0.895** | **0.967** |
| MLP | 0.885 | 0.827 | 0.855 | 0.940 |

The Random Forest is the best of the three. It is worth noting that tuning did not improve it: the baseline F1 was 0.897 against 0.895 tuned, a difference well inside the cross-validation spread of about 0.004, so the fairest reading is that the search found nothing better than the defaults for this model. Tuning did help the MLP, which went from 0.847 to 0.855.

## What is in the folder

```
exoplanet_detection.ipynb        the full analysis, run top to bottom
app.py                           Streamlit demo
data/                            the KOI catalogue
models/                          fitted pipelines saved by the notebook
results/                         metric tables and figures saved by the notebook
requirements.txt                 dependencies
test_dataset_20_percent.csv      the held-out test set, for the demo
test_false_positive_example.csv  a single false-positive row, for the demo
```

The saved models are whole pipelines rather than just the classifiers, so the imputer and scaler travel with them and the app cannot preprocess the data differently from the notebook.

## Notes and limitations

- Grouping CANDIDATE with CONFIRMED means that some of the positive labels are unverified. Training only on CONFIRMED would give cleaner labels, but it throws away a lot of data and answers a different question.
- 938 stars in the catalogue host more than one KOI, which is about a quarter of the rows, so the train/test split and the cross-validation folds are both grouped on `kepid`. A plain random split would let a model recognise the star instead of the transit.
- Every column produced by the vetting process is removed before training. `results/leakage_columns_removed.csv` lists each one and the reason it was excluded.
