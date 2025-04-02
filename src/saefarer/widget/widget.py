import os
import sqlite3
from pathlib import Path

import anywidget
import traitlets

import saefarer.analysis.database as db

_DEV = True


class Widget(anywidget.AnyWidget):
    if _DEV:
        _esm = "http://localhost:5173/js/widget.ts?anywidget"
        _css = ""
    else:
        bundled_assets_dir = Path(__file__).parent.parent / "static"
        _esm = bundled_assets_dir / "widget.js"
        _css = bundled_assets_dir / "style.css"

    height = traitlets.Int(0).tag(sync=True)

    model_info = traitlets.Dict().tag(sync=True)

    sae_ids = traitlets.List([]).tag(sync=True)
    sae_id = traitlets.Unicode().tag(sync=True)
    sae_data = traitlets.Dict().tag(sync=True)

    features = traitlets.List([]).tag(sync=True)

    feature_id = traitlets.Int(0).tag(sync=True)
    feature_data = traitlets.Dict().tag(sync=True)

    num_feature_table_rows = traitlets.Int(0).tag(sync=True)
    base_font_size = traitlets.Int(0).tag(sync=True)

    def __init__(self, path: str | os.PathLike, height: int = 600, **kwargs):
        super().__init__(**kwargs)

        path = Path(path)

        if not path.exists():
            raise OSError(f"Cannot read {path}")

        self.con = sqlite3.connect(path.as_posix())
        self.cur = self.con.cursor()

        self.height = height
        self.num_feature_table_rows = 10
        self.base_font_size = 16

        self.model_info = db.read_misc("model_info", self.cur)

        self.sae_ids = db.read_sae_ids(self.cur)
        self.sae_id = self.sae_ids[0]
        self.sae_data = db.read_sae_data(self.sae_ids[0], self.cur)

        self.feature_id = self.sae_data["alive_feature_ids"][0]
        self.feature_data = db.read_feature_data(self.feature_id, self.sae_id, self.cur)
        self.features = db.query_features(self.sae_id, self.cur)

    @traitlets.observe("feature_id")
    def _on_feature_id_change(self, change):
        new_feature_id = change["new"]
        new_feature_data = db.read_feature_data(new_feature_id, self.sae_id, self.cur)

        self.feature_id = new_feature_id
        self.feature_data = new_feature_data
