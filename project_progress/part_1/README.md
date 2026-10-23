# Part 1: Text Processing and Exploratory Data Analysis

Part 1 is implemented in [`part1.ipynb`](./part1.ipynb). The notebook loads the
fashion-products dataset, preprocesses its text fields, saves the processed
documents, and generates exploratory statistics, plots, and ranking tables.

## 1. Prepare the environment

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bat
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The notebook also imports `wordcloud`. If it is not already installed in your
environment, install it with:

```bash
python -m pip install wordcloud
```

Make sure the dataset exists at
`data/fashion_products_dataset.json` in the repository root.

## 2. Open and run the notebook

### Recommended: VS Code

1. Open the repository in VS Code.
2. Open [`part1.ipynb`](./part1.ipynb).
3. Select the `.venv` Python interpreter as the notebook kernel.
4. Run the notebook from top to bottom with **Run All**.

### Jupyter from a terminal

Run the notebook from its own directory. This is important because the
notebook loads the dataset using the relative path
`../../data/fashion_products_dataset.json`.

```bash
cd project_progress/part_1
python -m pip install jupyter
jupyter notebook part1.ipynb
```

In the Jupyter interface, select `part1.ipynb` and execute all cells in order.

The notebook writes the processed collection to
[`processed_docs.json`](./processed_docs.json) and saves generated figures in
[`eda_output/`](./eda_output/).

## 3. Text-processing functions

The first part of the notebook defines the preprocessing pipeline used for the
text fields:

1. `normalize(text)` converts text to lowercase and removes URLs.
2. `remove_punctuation(text)` replaces punctuation with spaces.
3. `remove_stopwords(tokens)` removes English stop words using NLTK.
4. `drop_short(tokens)` removes single-letter tokens, while keeping numbers.
5. `stem(tokens)` applies the NLTK `PorterStemmer`.
6. `build_terms(text)` calls the previous steps in the complete pipeline.

To inspect or change the preprocessing behavior, edit these functions in the
preprocessing cell and run that cell and all following cells again. The
resulting terms are stored in the processed fields such as `title_terms`,
`description_terms`, and `all_terms`.

## 4. Selecting ranking scores and analysis options

There is no command-line menu or separate ranking algorithm selector. The
ranking choices are made by editing and running the corresponding notebook
cells. Part 1 does not implement a search-retrieval score such as TF-IDF,
BM25, or PageRank; its rankings are exploratory rankings of product fields and
category frequencies.

### Product rankings

The final ranking cell uses `sort_values()` and displays the first 10 products.
The three available rankings are:

```python
df.sort_values("average_rating", ascending=False).head(10)
df.sort_values("selling_price", ascending=False).head(10)
df.sort_values("discount", ascending=False).head(10)
```

To select a different ranking:

1. Find the final cell containing `Top 10 by rating`.
2. Change the column passed to `sort_values()`:
   - `average_rating` for the rating ranking.
   - `selling_price` for the most expensive products.
   - `discount` for the largest discount.
   - Any other numeric column, such as `actual_price`, for another ranking.
3. Keep `ascending=False` for highest values first, or set
   `ascending=True` for lowest values first.
4. Change `.head(10)` to another number to select a different number of
   results, for example `.head(20)`.
5. Run the edited cell.

The displayed columns are controlled by `show_cols`. Add or remove field names
there if more or fewer product details should be shown.

### Categorical rankings and plots

The notebook also creates frequency rankings for categorical fields. In the
cell that loops over the categorical columns, change the list of columns to
choose which categories to analyze, for example:

```python
fields = ["brand", "category", "sub_category", "seller"]
```

The `n` argument controls how many values are displayed in each frequency
ranking. Change it when calling `value_counts().head(n)` if you want more or
fewer categories.

Numeric distributions are selected in the cell containing:

```python
for col in ["selling_price", "actual_price", "discount", "average_rating"]:
```

Add or remove column names in that list to choose which histograms are
generated. Price histograms use a logarithmic y-axis because prices are
skewed.

## 5. Re-running after changes

After changing preprocessing functions, ranking columns, or plot options, use
**Restart Kernel and Run All** (or restart the Jupyter kernel and execute every
cell from the beginning). This ensures that `processed`, `df`, and all saved
outputs are rebuilt consistently.
