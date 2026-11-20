# Part 3: Ranking & Filtering

Part 3 is implemented in [`part3.ipynb`](./part3.ipynb). It uses the documents
processed in Part 1 (`../part_1/processed_docs.json`) and the relevance labels
of Part 2 (`../part_2/test_labels.csv`).

## 1. Prepare the environment

Word2vec uses gensim, which has no version for Python 3.14 yet, so this part
needs Python 3.11 (3.12 also works). From the repository root:

```bash
py -3.11 -m venv .venv
```

```bat
.venv\Scripts\activate
```

```bash
python -m pip install nltk numpy pandas gensim ipykernel
```

## 2. Run the notebook

Open [`part3.ipynb`](./part3.ipynb) in VS Code, select the `.venv` environment
as the kernel and use **Run All**. Training word2vec takes about 20 seconds.

## 3. Files

- `myapp/search/algorithms.py`: besides the Part 2 functions, it contains
  `create_bm25_index` and `score_bm25` (BM25), `score_custom` (our score),
  `train_word2vec`, `text_vector` and `score_word2vec` (word2vec + cosine), and
  `rank`, which sorts the scores and breaks ties by pid.
- `part3.ipynb`: rankings of the five test queries with the four methods, the
  word2vec top 20 lists and the comparison with the evaluation metrics.

## 4. Changing the ranking

- BM25 parameters: `score_bm25(..., k1=1.2, b=0.75)`.
- Our score: the weights of text, title, rating and discount are the argument
  `weights=(0.5, 0.3, 0.15, 0.05)` of `score_custom`, and the out-of-stock
  penalty is `stock_penalty=0.5`.
- Word2vec: the vector size and the seed are arguments of `train_word2vec`.

Run all the cells again after any change.
