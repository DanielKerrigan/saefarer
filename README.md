# SAEfarer

## Development installation

Create a virtual environment and and install saefarer in _editable_ mode with the
optional development dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

You then need to install the JavaScript dependencies and run the development server.

```sh
npm install
npm run dev
```

Open `example.ipynb` in JupyterLab, VS Code, or your favorite editor
to start developing. Changes made in `js/` will be reflected
in the notebook.

## References

This project uses code from the following repositories:

- [OpenAI's sparse_autoencoder](https://github.com/openai/sparse_autoencoder)
- [EleutherAI's sae](https://github.com/EleutherAI/sae)
- [SAELens](https://github.com/jbloomAus/SAELens)
- [1L-Sparse-Autoencoder](https://github.com/neelnanda-io/1L-Sparse-Autoencoder)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [sae_vis](https://github.com/callummcdougall/sae_vis)
- [SAEDashboard](https://github.com/jbloomAus/SAEDashboard)
