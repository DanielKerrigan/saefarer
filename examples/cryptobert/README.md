# CryptoBERT

<a target="_blank" href="https://colab.research.google.com/github/DanielKerrigan/saefarer/blob/main/examples/cryptobert/colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This example analyzes the [CryptoBERT](https://huggingface.co/ElKulako/cryptobert) model, which does sentiment classification of social media posts about cryptocurrencies.

If you are in a Python environment that has SAEfarer installed, then running the example from scratch would look like this:

```bash
# 1. download and tokenize the dataset
python download-stocktwits-crypto.py
python tokenize-stocktwits-crypto.py

# 2. train the sparse autoencoder
python train.py

# 3. precompute information about SAE for widget
python analyze.py
```

Then you can run the `widget.ipynb` notebook to use the tool.

Training and analyzing the SAE can take a while. If you would just like to try the tool without having to do that yourself, then we provide a notebook that you can run in [Google Colab](https://colab.research.google.com/github/DanielKerrigan/saefarer/blob/main/examples/cryptobert/colab.ipynb). This notebook installs SAEfarer, downloads an already trained SAE and its precomputed data, and runs the widget.
