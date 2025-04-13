import math
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import anywidget
import traitlets

import saefarer.analysis.database as db

if TYPE_CHECKING:
    from saefarer.analysis.types import RankingOption
    from saefarer.widget.config import WidgetConfig

_DEV = True


class Widget(anywidget.AnyWidget):
    if _DEV:
        _esm = "http://localhost:5173/js/widget.ts?anywidget"
        _css = ""
    else:
        bundled_assets_dir = Path(__file__).parent.parent / "static"
        _esm = bundled_assets_dir / "widget.js"
        _css = bundled_assets_dir / "saefarer.css"

    height = traitlets.Int().tag(sync=True)
    base_font_size = traitlets.Int().tag(sync=True)
    n_table_rows = traitlets.Int().tag(sync=True)

    model_info = traitlets.Dict().tag(sync=True)

    sae_ids = traitlets.List().tag(sync=True)
    sae_id = traitlets.Unicode().tag(sync=True)
    sae_data = traitlets.Dict().tag(sync=True)

    table_ranking_option = traitlets.Dict().tag(sync=True)  # type: ignore
    table_page_index = traitlets.Int().tag(sync=True)
    max_table_page_index = traitlets.Int().tag(sync=True)
    table_features = traitlets.List().tag(sync=True)

    detail_feature = traitlets.Dict().tag(sync=True)
    detail_feature_id = traitlets.Int().tag(sync=True)

    def __init__(self, path: str | os.PathLike, cfg: "WidgetConfig", **kwargs):
        super().__init__(**kwargs)

        path = Path(path)

        if not path.exists():
            raise OSError(f"Cannot read {path}")

        self.con = sqlite3.connect(path.as_posix())
        self.cur = self.con.cursor()

        self.height = cfg.height
        self.base_font_size = cfg.base_font_size
        self.n_table_rows = cfg.n_table_rows

        self.model_info = db.read_misc("model_info", self.cur)

        self.sae_ids = db.read_sae_ids(self.cur)
        self.sae_id = self.sae_ids[0]
        self.sae_data = db.read_sae_data(self.sae_ids[0], self.cur)

        self.table_ranking_option: "RankingOption" = {
            "kind": "feature_id",
            "descending": True,
        }
        self.table_page_index = 0
        self.max_table_page_index = (
            math.ceil(self.sae_data["n_alive_features"] / self.n_table_rows) - 1
        )
        self.table_features = db.rank_features(
            self.sae_id,
            self.cur,
            self.table_ranking_option,
            self.table_page_index,
            self.n_table_rows,
            len(self.model_info["labels"]),
        )

        self.detail_feature = self.table_features[0]
        self.detail_feature_id = self.detail_feature["feature_id"]

    @traitlets.observe("detail_feature_id")
    def _on_detail_feature_id_change(self, change):
        new_feature_id = change["new"]

        # this happens when we reset detail_feature_id to the id of
        # detail_feature when an invalid value is passed
        if new_feature_id == self.detail_feature["feature_id"]:
            return

        for feature in self.table_features:
            if new_feature_id == feature["feature_id"]:
                self.detail_feature = feature
                return

        feature = db.read_feature_data(new_feature_id, self.sae_id, self.cur)

        if feature is not None:
            self.detail_feature = feature
            return

        self.detail_feature_id = self.detail_feature["feature_id"]

    @traitlets.observe("table_page_index")
    def _on_table_page_index_change(self, _):
        self.table_features = db.rank_features(
            self.sae_id,
            self.cur,
            self.table_ranking_option,
            self.table_page_index,
            self.n_table_rows,
            len(self.model_info["labels"]),
        )

    @traitlets.observe("table_ranking_option")
    def table_ranking_option_change(self, _):
        """When the ranking option is changed, go back to the first page.
        Updating table_features will happen in the change handler for
        table_page_index. If we are already on the first change,
        then update table_features here."""

        if self.table_page_index == 0:
            self.table_features = db.rank_features(
                self.sae_id,
                self.cur,
                self.table_ranking_option,
                self.table_page_index,
                self.n_table_rows,
                len(self.model_info["labels"]),
            )
        else:
            self.table_page_index = 0
