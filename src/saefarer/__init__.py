import importlib.metadata
import os
from pathlib import Path
from typing import Union

import anywidget
import traitlets

from saefarer.widget_utils import convert_keys_to_ints, read_feature_data, read_json

try:
    __version__ = importlib.metadata.version("saefarer")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


class Widget(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "static" / "widget.js"
    _css = Path(__file__).parent / "static" / "style.css"

    height = traitlets.Int(0).tag(sync=True)

    sae_data = traitlets.Dict().tag(sync=True)
    feature_index = traitlets.Int().tag(sync=True)
    feature_data = traitlets.Dict().tag(sync=True)

    def __init__(self, root_dir: Union[str, os.PathLike], height: int = 600, **kwargs):
        super().__init__(**kwargs)

        root_dir = Path(root_dir)

        if not root_dir.exists():
            raise OSError(f"Cannot read {root_dir}")

        self.root_dir = root_dir
        self.height = height

        self.sae_data = read_json(root_dir / "overview.json")
        self.sae_data["feature_index_to_path"] = convert_keys_to_ints(
            self.sae_data["feature_index_to_path"]
        )
        self.feature_index = self.sae_data["alive_feature_indices"][0]
        self.feature_data = read_feature_data(
            self.feature_index, self.sae_data, self.root_dir
        )

    @traitlets.observe("feature_index")
    def _on_feature_index_change(self, change):
        new_feature_index = change["new"]
        new_feature_data = read_feature_data(
            new_feature_index, self.sae_data, self.root_dir
        )

        self.feature_index = new_feature_index
        self.feature_data = new_feature_data

