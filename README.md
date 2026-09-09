# Bitcoin Ransomware Transaction Classifier
 
Predicting whether a Bitcoin payment is associated with ransomware using transaction-level metadata — no wallet ownership or address-history data required.
 
**Team:** Mario Valek, Siddhant Vashisht, Edwin Mui
 
## Overview
 
Bitcoin's pseudonymity makes it a preferred payment rail for ransomware operators, but transaction *patterns* still leave a fingerprint. Using the [BitcoinHeist Ransomware Dataset](https://www.kaggle.com/datasets/sapere0/bitcoinheist-ransomware-dataset) (~2.9M labeled Bitcoin transactions, 2011–2018), this project builds and compares supervised and unsupervised models that classify a transaction as ransomware vs. legitimate ("white") based on six raw features: `length`, `weight`, `count`, `looped`, `neighbors`, and `income`.
 
## Pipeline
 
1. **Data loading & profiling** — loaded ~2.9M rows in pandas and Polars side by side to benchmark preprocessing performance (Polars cut wall-clock time roughly in half on groupby/aggregation operations).
2. **EDA** — label distribution, CDFs of each transaction feature, a correlation heatmap, PCA projection, and a time-series view of ransomware activity from 2011–2018 to surface class imbalance and outlier structure up front.
3. **Feature engineering** — derived `complexity_score` (length × looped) and `size_ratio` (weight / count) to capture transaction shape beyond the raw fields.
4. **Class imbalance handling** — the raw data is ~93% legitimate transactions. Addressed via resampling to a 1:2 (ransomware:white) ratio (~124K rows) rather than relying on accuracy alone, which is a misleading metric under this skew.
5. **Modeling** — trained and compared three approaches on the balanced dataset:
   - **XGBoost** (interpretable, feature-importance-driven) — baseline, then re-tuned via `RandomizedSearchCV` over depth, learning rate, subsampling, and regularization, optimizing for ROC-AUC under 5-fold CV.
   - **k-Nearest Neighbors** (supervised, distance-based) — sanity-check against a fundamentally different model family.
   - **K-Means clustering** (unsupervised) — tested whether ransomware/legitimate transactions separate without labels at all.
## Results
 
| Model | Setup | Accuracy |
|---|---|---|
| XGBoost | Baseline, imbalanced data | 66% |
| XGBoost | Balanced data (1:2 resampling) | 69% |
| XGBoost | Balanced + hyperparameter-tuned (RandomizedSearchCV, 5-fold CV) | **88%** |
| k-NN | Balanced data, k=5 | 81% |
| K-Means | Unsupervised, 2 clusters | Did not separate classes |
 
- The tuned XGBoost model correctly identified ~88.8% of actual ransomware transactions and ~87.4% of legitimate ones on held-out data — a large improvement over the untuned baseline's 97% false-positive rate on ransomware predictions.
- **`income` was the single most important feature** (importance score 0.29), with `looped` (transaction self-referencing/circularity) a distant second (0.16) — suggesting ransomware payments are distinguishable primarily by transaction value patterns rather than network topology.
- K-Means clustering failed to recover the true label structure, indicating the classes aren't linearly separable in the raw/PCA feature space without supervision — a useful negative result that motivated sticking with supervised approaches.
## Key takeaways
 
- Naive accuracy is actively misleading on this dataset; F1/precision/recall and confusion-matrix analysis were necessary to catch a model that was effectively predicting "always legitimate."
- Hyperparameter tuning delivered the single largest accuracy gain (69% → 88%) of any step in the pipeline — larger than either feature engineering or resampling alone.
- Simple, engineered features (`income`, `looped`) outperformed more complex network-derived features (`neighbors`, `network_activity`) in the final model, which is a useful signal for keeping future iterations of this pipeline lean.
## Limitations & future work
 
- Dataset only covers activity through 2018; ransomware tactics and typical transaction shapes have shifted significantly since.
- Bitcoin addresses don't map 1:1 to real-world entities, so any "per-address" pattern is a proxy at best.
- Compute constraints limited hyperparameter search breadth and ruled out heavier models (e.g., deep learning) within project scope.
- A natural next step is a streaming/real-time scoring pipeline that could flag suspicious transactions as they're broadcast, rather than in batch after the fact.
## Stack
 
`pandas` · `polars` · `scikit-learn` · `XGBoost` · `matplotlib` / `seaborn` · `statsmodels`
