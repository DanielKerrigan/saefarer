import os
import sqlite3
from pathlib import Path
from typing import Union

import anywidget
import traitlets

import saefarer.database as db


class Widget(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "static" / "widget.js"
    _css = Path(__file__).parent / "static" / "style.css"

    height = traitlets.Int(0).tag(sync=True)

    sae_ids = traitlets.List([]).tag(sync=True)

    sae_id = traitlets.Unicode().tag(sync=True)
    feature_id = traitlets.Int().tag(sync=True)

    sae_data = traitlets.Dict().tag(sync=True)
    feature_data = traitlets.Dict().tag(sync=True)

    def __init__(self, path: Union[str, os.PathLike], height: int = 600, **kwargs):
        super().__init__(**kwargs)

        path = Path(path)

        if not path.exists():
            raise OSError(f"Cannot read {path}")

        self.con = sqlite3.connect(path.as_posix())
        self.cur = self.con.cursor()

        self.height = height

        self.sae_ids = db.read_sae_ids(self.cur)
        self.sae_id = self.sae_ids[0]
        self.sae_data = db.read_sae_data(self.sae_ids[0], self.cur)

        self.feature_id = self.sae_data["alive_feature_ids"][0]
        self.feature_data = db.read_feature_data(self.feature_id, self.sae_id, self.cur)

    @traitlets.observe("feature_id")
    def _on_feature_id_change(self, change):
        new_feature_id = change["new"]
        new_feature_data = db.read_feature_data(new_feature_id, self.sae_id, self.cur)

        self.feature_id = new_feature_id
        self.feature_data = new_feature_data
