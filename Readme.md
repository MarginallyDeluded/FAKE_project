# Fake News Classification & Feature Processing, and LDA on WELFake dataset


A systematic classical machine learning and NLP benchmark on the **WELFake** news dataset (`72,070` articles), evaluating how lexical, stylometric, topical, and combined feature representations impact classifier geometry, generalization, and susceptibility to dataset artifacts.

Mini Project Assignment SD:-10.11.2026 ED:-25.11.2026

For Course: Introduction to Machine Learning

Course Code: PCCAIML 502

## Overview & Motivation

Rather than relying on a single black-box model, this project implements a **four-track feature ablation framework** inspired by comparative misinformation benchmarks (including the $k = \lfloor\sqrt{N_{\text{train}}}\rfloor$ heuristic of Lall & Sharma). Each feature representation is routed to classifiers matched to its dimensionality and statistical properties, followed by learning-curve diagnostics and linear weight auditing to separate genuine linguistic signals from scraping shortcuts.

### Key Highlights

* **Peak Performance:** `LinearSVC` trained on `100,000` TF-IDF unigrams and bigrams achieves **`0.9750` F1-score** (`97.42%` accuracy) across `14,414` held-out test articles.

* **Leak-Free Serialization Pipeline:** Vectorizers, scalers, and topic models are fitted strictly on the training split (`57,656` rows) and persisted as compressed sparse/dense artifacts (`scipy.sparse`, `NumPy`, `joblib`).

* **Unsupervised Topical Profiling & Auditing:** A 12-component Latent Dirichlet Allocation (LDA) pipeline captures latent political/policy themes and supplies dense semantic vectors for $k$-NN benchmarking.

* **Diagnostic Error & Artifact Auditing:** Coefficient inspection uncovers specific CMS boilerplate, embedded social media remnants, and 2016–2017 temporal collection bias contributing to high linear separability.

## Dataset & Preprocessing Pipeline

The cleaned corpus comprises **`72,070` news articles** (`title` + `text`) split into stratified training and test partitions:

* **Training Set:** `57,656` documents (`80%`)

* **Test Set:** `14,414` documents (`20%` — `7,006` Class `0` / `7,408` Class `1`)

### Core Preprocessing & Leak Prevention

1. **Regex Leak Scrubbing:** Explicit publisher tags (e.g., `reuters`, `breitbart`, `21wire`) are stripped prior to feature extraction, and empty records are removed.

2. **Strict Train-Only Fitting:** All stateful transformers (`TfidfVectorizer`, `StandardScaler`, `CountVectorizer`, `LatentDirichletAllocation`) are fitted exclusively on training folds before calling `.transform()` on test data.

3. **Compressed Persistence (`norm_data/`):**

   * Sparse CSR matrices (`X_train_tfidf.npz`, `X_train_num.npz`, `X_train_final.npz`) via `scipy.sparse.save_npz`

   * Dense arrays (`X_train_topics.npy`, `y_train.npy`) via `numpy.save`

   * Fitted transformers (`tfidf_vectorizer.joblib`, `num_scaler.joblib`, `lda_model.joblib`) via `joblib.dump`

## Feature Engineering Tracks

| Track | Feature Representation | Dimensions | Format | Target Model Families | 
| ----- | ----- | ----- | ----- | ----- | 
| **Track A** | **Stylometric Features** (`title_len`, `text_len`, `excl_count`, `quest_count`, `caps_words`) | `5` | Dense (`StandardScaler`) | `LinearSVC`, `LogisticRegression`, `DecisionTree`, `AdaBoost` | 
| **Track B** | **Latent Dirichlet Allocation (LDA)** topic probability distributions | `12` | Dense (`float64`) | $k$-Nearest Neighbors ($k = \lfloor\sqrt{57656}\rfloor = 241$) | 
| **Track C** | **TF-IDF N-Grams** (unigrams + bigrams, non-negative) | `100,000` | Sparse CSR | `MultinomialNB`, `LinearSVC`, `LogisticRegression` | 
| **Track D** | **Combined Representation** (`TF-IDF` $\oplus$ `Stylometric`) | `100,005` | Sparse CSR (`hstack`) | `LinearSVC`, `LogisticRegression` | 

## LDA Preprocessing & Exploratory Topic Modeling

To replace proprietary semantic lexicons (such as Empath) with an open, data-driven thematic representation, an unsupervised **12-topic Latent Dirichlet Allocation (LDA)** pipeline was built both to inspect corpus structure and to generate dense document-topic probability vectors for **Track B**.

### LDA-Specific Text Preprocessing

1. **Raw Count Vectorization (`CountVectorizer`):** Because LDA's generative Dirichlet-multinomial model operates on discrete word frequencies rather than sublinear weights, articles are tokenized into raw term-count matrices rather than TF-IDF vectors.

2. **Domain Stopword & Frequency Filtering:**

   * `stop_words=final_stoplist`: Removes standard English stopwords alongside high-frequency journalistic filler words.

   * `max_df=0.85` & `min_df=20`: Prunes overly ubiquitous terms appearing in more than `85%` of documents as well as rare typos/tokens appearing in fewer than `20` documents.

   * `token_pattern=r"\b[a-z0-9]{3,}\b"`: Restricts the vocabulary to alphanumeric tokens of length $\ge 3$, stripping stray punctuation and 1–2 character fragments.

3. **Train/Test Isolation:** `lda_vect_train` and `LatentDirichletAllocation(n_components=12, learning_method="online", batch_size=2048, max_iter=15, random_state=42)` are fitted strictly on `X_train_txt` (`57,656` documents) and applied via `.transform()` to `X_test_txt` (`14,414` documents), saving `(57656, 12)` and `(14414, 12)` probability distributions to `norm_data/`.

### Intertopic Distance Map & Corpus Insights

* **Well-Separated Latent Clusters:** Projected onto two principal components (`PC1` vs. `PC2`) via multidimensional scaling (`pyLDAvis`), the 12 topics spread cleanly across all four quadrants with minimal overlap (only a minor intersection between Topics `4` and `6`) and balanced marginal topic proportions.

* **Dominant Political & Policy Discourse:** The Top-30 most salient terms (`trump`, `president`, `clinton`, `state`, `obama`, `government`, `house`, `donald`, `campaign`, `election`, `russia`, `fbi`, `investigation`) confirm that the corpus centers heavily on 2016–2017 U.S. presidential politics, governance, and national security.

* **Classification Takeaway:** While these 12 topical distributions allow $k$-NN ($k = 241$) to reach `0.7848` F1—outperforming linear stylometric models (`0.7541–0.7588` F1)—both real and fake outlets cover the same political subjects. Distinguishing reliable reporting from fabrication on identical topics requires the fine-grained lexical and framing cues captured in **Track C** (`0.9750` F1).

## Benchmark Results (`14,414` Test Samples)

| \# | Model | Feature Track | Accuracy | Precision | Recall | F1-Score | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| 0 | `LinearSVC` | Stylometric (`5`) | `0.7689` | `0.8322` | `0.6894` | `0.7541` | 
| 1 | `LogisticRegression` | Stylometric (`5`) | `0.7688` | `0.8178` | `0.7077` | `0.7588` | 
| 2 | `DecisionTree` | Stylometric (`5`) | `0.8153` | `0.8427` | `0.7877` | `0.8143` | 
| 3 | `AdaBoost (35 trees)` | Stylometric (`5`) | `0.7908` | `0.8140` | `0.7685` | `0.7906` | 
| 4 | `k-NN (k=241)` | LDA Topics (`12`) | `0.7772` | `0.7794` | `0.7902` | `0.7848` | 
| 5 | `MultinomialNB` | TF-IDF N-grams (`100k`) | `0.8762` | `0.8670` | `0.8967` | `0.8816` | 
| **6** | **`LinearSVC` (Best)** | **TF-IDF N-grams (`100k`)** | **`0.9742`** | **`0.9691`** | **`0.9811`** | **`0.9750`** | 
| 7 | `LogisticRegression` | TF-IDF N-grams (`100k`) | `0.9566` | `0.9517` | `0.9646` | `0.9581` | 
| 8 | `LinearSVC` | Combined (`100k + 5`) | `0.9729` | `0.9672` | `0.9806` | `0.9739` | 
| 9 | `LogisticRegression` | Combined (`100k + 5`) | `0.9576` | `0.9532` | `0.9649` | `0.9590` | 

### Best Model Classification Report (`LinearSVC` on `TF-IDF N-grams`)

| Class | Precision | Recall | F1-Score | Support | 
| ----- | ----- | ----- | ----- | ----- | 
| **Class `0`** | `0.9798` | `0.9669` | `0.9733` | `7,006` | 
| **Class `1`** | `0.9691` | `0.9811` | `0.9750` | `7,408` | 
| **Accuracy** | — | — | **`0.9742`** | **`14,414`** | 
| **Macro Avg** | `0.9744` | `0.9740` | `0.9742` | `14,414` | 
| **Weighted Avg** | `0.9743` | `0.9742` | `0.9742` | `14,414` | 

* **Confusion Matrix Summary:** Out of `14,414` test articles, `svc_tfidf` makes only `372` total errors (`2.58%` error rate): `6,774` True Negatives, `232` False Positives, `140` False Negatives, and `7,268` True Positives.

## Diagnostic Analysis & Key Takeaways

### 1. Model Geometry Must Match Feature Geometry

* **Low-Dimensional Dense Space (`5` features):** Non-linear `DecisionTree` (`0.8143` F1) outperforms linear hyperplanes (`LinearSVC` `0.7541`, `LogisticRegression` `0.7588`) because surface formatting cues interact non-linearly (e.g., short headline length combined with uppercase bursts and exclamation marks).

* **High-Dimensional Sparse Space (`100,000` features):** Discriminative linear classifiers (`LinearSVC` `0.9750`, `LogisticRegression` `0.9581`) outperform `MultinomialNB` (`0.8816`) by nearly `+9.3` F1 points, as Naive Bayes' conditional independence assumption is violated by overlapping unigrams and bigrams.

### 2. Feature Ablation (`TF-IDF` vs. `Combined`)

Appending the 5 standardized stylometric columns onto `100,000` L2-normalized TF-IDF features yields no practical gain (`0.9750` $\rightarrow$ `0.9739` F1 for `LinearSVC`). Weight inspection of `svc_final` confirms that stylometric coefficients (`caps_words`: `+0.2047`, `title_len`: `+0.1259`, `excl_count`: `+0.1058`, `quest_count`: `-0.0281`, `text_len`: `-0.0133`) are over $40\times$ smaller in magnitude than top lexical weights (`via`: `+9.12`, `said`: `-5.66`), while mixing dense z-scores with sparse unit-norm vectors degrades `liblinear` coordinate descent conditioning.

### 3. 5-Fold Cross-Validation Learning Curves

* **Track A (`LinearSVC` on Stylometric):** Exhibits classic **high bias (underfitting)**—both Train and Validation F1 plateau immediately around `~0.75` from `10%` ($\approx 4,600$ rows) to `100%` ($\approx 46,100$ rows) with near-zero generalization gap, proving that additional training rows cannot overcome a 5-feature bottleneck.

* **Track C (`LinearSVC` on TF-IDF):** Operates in a well-regularized $p > n$ regime—Train F1 remains at `1.00` while Validation F1 climbs steadily from `~0.939` to `~0.974` with tight cross-validation variance bands.

### 4. Linear Coefficient Audit & Dataset Shortcut Discovery

Inspecting the highest-magnitude weights in `LinearSVC` reveals three distinct signal sources behind the `97.42%` accuracy:

* **Genuine Journalistic Style:** Attribution verbs (`said` `-5.66`, `said on` `-3.79`) and weekday datelines (`on monday` through `on friday`) strongly anchor Class `0`, whereas urgency framing (`breaking` `+2.66`) pushes toward Class `1`.

* **CMS & Social Media Artifacts:** Photo credits and template footers (`via` `+9.12`, `read more` `+4.36`, `featured image` `+4.07`, `image via` `+3.24`, `getty images` `+2.53`) act as strong Class `1` shortcuts, while embedded tweet and newspaper remnants (`follow` `-4.40`, `on twitter` `-4.02`, `pic twitter` `-2.83`, `york times` `-2.65`) pull toward Class `0`.

* **Temporal Collection Bias:** Election-window tokens (`october` `+4.52`, `november` `+3.71`, `hillary` `+2.75`) correlate with Class `1`, whereas post-inauguration phrasing (`president donald` `-4.09`) correlates with Class `0`.

## Repository Structure

```
├── notebooks/
│   ├── norm_data/                    # Serialized train/test matrices (.npz, .npy) & transformers (.joblib)
│   ├── WELFake_Dataset.csv           # Raw WELFake dataset (72,070 cleaned news articles)
│   ├── embedding.ipynb               # Regex cleaning, train/test split, TF-IDF & stylometric extraction
│   ├── LDA.ipynb                     # CountVectorization, 12-topic LDA fitting & pyLDAvis visualization
│   └── model_eval.ipynb              # Tracks A–D training, evaluation, learning curves & weight audit
├── assets/
│   ├── lda_topic_map.png        # pyLDAvis intertopic distance map & top-30 salient terms
│   ├── conf_matrix_svc.png      # Held-out test confusion matrix for svc_tfidf
│   └── learning_curves.png           # 5-fold CV sample-size sweep (Track A vs. Track C)
└── README.md

```
## Author

* **Raajjyoti_Deb** . Roll no. '35530824047' . GKCIET, Malda, West Bengal, B-Tech CSE(AI-ML)

## Next Steps

* \[ \] Run an **artifact-scrubbed ablation** removing CMS boilerplate (`featured image`, `read more`, `via`, `getty`) and social remnants (`pic twitter`, `https`) to measure pure linguistic generalization.

* \[ \] Add threshold-independent **ROC-AUC curves** using `LinearSVC.decision_function()`.

* \[ \] Benchmark contextual transformer embeddings (e.g., DistilBERT / RoBERTa) against the `0.9750` F1 `LinearSVC` baseline.
