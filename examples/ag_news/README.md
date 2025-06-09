# CryptoBERT

<a target="_blank" href="https://colab.research.google.com/github/DanielKerrigan/saefarer/blob/main/examples/ag_news/colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This example analyzes a [model](https://huggingface.co/Kyle1668/ag-news-19200-bert-base-uncased) trained to categorize articles from the [AG News](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) dataset as either World, Sports, Business, or Sci/Tech.

If you are in a Python environment that has SAEfarer installed, then running the example from scratch would look like this:

```bash
# 1. download and tokenize the dataset
python download-dataset.py
python tokenize-dataset.py

# 2. train the sparse autoencoder
python train.py

# 3. precompute information about SAE for widget
python analyze.py
```

Then you can run the `widget.ipynb` notebook to use the tool.

Training and analyzing the SAE can take a while. If you would just like to try the tool without having to do that yourself, then we provide a notebook that you can run in [Google Colab](https://colab.research.google.com/github/DanielKerrigan/saefarer/blob/main/examples/ag_news/colab.ipynb). This notebook installs SAEfarer, downloads an already trained SAE and its precomputed data, and runs the widget.
