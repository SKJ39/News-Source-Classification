# News-Source-Classification

CIS 5190 Project B: classify a news article as **Fox News (0)** or **NBC News (1)** from the headline slug in its URL.

**Final submission: the ensemble.** It takes a weighted average of the probabilities from three models: TF-IDF LR, TextCNN and BagMLP, with weights 0.65 / 0.10 / 0.25. The submission files are in `data/ensemble/submission/`.

| Model | 5-fold CV | Local TEST (934 rows) |
|---|---|---|
| TF-IDF + LR | 0.8605 | 0.8576 |
| TextCNN | 0.8349 | 0.8383 |
| BagMLP | 0.8375 | 0.8415 |
| **Ensemble** | **0.8646** (OOF) | 0.8533 |

On local TEST, LR and the ensemble differ by only 4 rows, which is within noise. The ensemble scores higher on the leaderboard, so it is the one we submit.

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | `prepare_data` used for submission: turns each URL into its slug. It does no filtering, dedup or reordering. |
| `model_nn.py` | Ensemble model. Rename it to `model.py` when submitting. |
| `model_tfidf.py` | Single LR model, kept as an alternative. |
| `scrape_matched.py` | Scrapes the Fox / NBC sitemaps and writes `data/scraped_matched.csv`. |
| `data_split.py` | Data cleaning and splitting shared by all training scripts. |
| `build_data.py` | Old data-building script. Only its slug and label helper functions are still used. |
| `train_tfidf.py` | LR: picks hyperparameters with 5-fold CV, evaluates once on TEST, then retrains on all data and exports. |
| `train_nn.py` | TextCNN / BagMLP: picks hyperparameters with 5-fold CV and writes OOF probabilities. |
| `ensemble.py` | Searches mix weights on the OOF probabilities, evaluates once on TEST, then retrains on all data and exports. |
| `eval_project_b.py`, `baseline_model.py` | Evaluation script and baseline model provided by the course. |

All outputs go under `data/`, in `tfidf/`, `nn/{cnn,bag}/` and `ensemble/`. Each of these directories contains:
- `cv_results.csv`: the CV score of every hyperparameter setting tried.
- `submission/`: a folder that is ready to submit.

`data/raw_sitemaps/` is the scraper cache and is listed in .gitignore.

## How to run

```bash
python scrape_matched.py                              # scrape (fast when cached)
python train_tfidf.py --scraped data/scraped_matched.csv
python train_nn.py --arch cnn                         # slow on CPU
python train_nn.py --arch bag                         # very slow on CPU (hours)
python ensemble.py --scraped data/scraped_matched.csv

# Reproduce the local TEST score (0.8533)
# best_trainval.pt is not in git (35MB); ensemble.py regenerates it.
python eval_project_b.py --model model_nn.py --preprocess preprocess.py \
    --csv data/ensemble/test.csv --weights data/ensemble/best_trainval.pt
```

Note: `submission/model.pt` is trained on all data, including TEST. Any score it gets on TEST or `url_train.csv` is inflated. Use it only to check that the submission loads and runs.

## Data split (`data_split.py`, seed 42)

1. `url_train.csv` has 4800 rows. After dropping rows that are unlabeled, have an empty slug or repeat an earlier slug, **4670** remain.
2. A stratified 20% (934 rows) is held out as **TEST** and evaluated only once, at the end.
3. The remaining 3736 rows (**TRAINVAL**) are split into 5 folds. Each fold trains on about 2989 rows and validates on about 747.
4. Scraped rows are added on the training side only. They are deduplicated by slug against all of `url_train` (including TEST) and against each other, leaving **46,178** rows.

Validation and TEST contain only `url_train` rows, because the leaderboard test set comes from the same distribution as `url_train`. Every hyperparameter, including the scraped-data weight `scraped_w`, is chosen by CV.

**How this differs from the original pipeline.** The original pipeline scraped recent news, mixed it with `url_train` and made one random 80/20 split. It had no dedup and no separate test set.

The new pipeline keeps only the sitemap shards from the same time period as `url_train`: 24 of the 540 candidates. LR CV accuracy under each setup:

| Training data | LR CV accuracy |
|---|---|
| `url_train` only | 0.8084 |
| `url_train` + old scraped data | 0.7883 |
| `url_train` + new scraped data | 0.8605 |
