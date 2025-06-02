import math
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import anywidget
import traitlets

import saefarer.analysis.database as db
from saefarer.analysis.config import AnalysisConfig
from saefarer.analysis.inference import inference

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from saefarer import sae
    from saefarer.analysis.types import (
        FeatureTokenSequence,
        InferenceInput,
        RankingOption,
    )
    from saefarer.widget.config import WidgetConfig

_DEV = False


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

    dataset_info = traitlets.Dict().tag(sync=True)
    model_info = traitlets.Dict().tag(sync=True)

    sae_ids = traitlets.List().tag(sync=True)
    sae_id = traitlets.Unicode().tag(sync=True)
    sae_data = traitlets.Dict().tag(sync=True)

    table_ranking_option = traitlets.Dict().tag(sync=True)  # type: ignore
    table_min_act_rate = traitlets.Float().tag(sync=True)
    table_page_index = traitlets.Int().tag(sync=True)
    max_table_page_index = traitlets.Int().tag(sync=True)
    num_filtered_features = traitlets.Int().tag(sync=True)
    table_features = traitlets.List().tag(sync=True)

    detail_feature = traitlets.Dict().tag(sync=True)
    detail_feature_id = traitlets.Int().tag(sync=True)

    can_inference = traitlets.Bool().tag(sync=True)
    inference_input = traitlets.Dict().tag(sync=True)  # type: ignore
    inference_output = traitlets.Dict().tag(sync=True)  # type: ignore

    def __init__(
        self,
        path: str | os.PathLike,
        cfg: "WidgetConfig",
        model: "PreTrainedModel | None" = None,
        tokenizer: "PreTrainedTokenizer | None" = None,
        sae: "sae.SAE | None" = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        path = Path(path)

        if not path.exists():
            raise OSError(f"Cannot read {path}")

        self.con = sqlite3.connect(path.as_posix())
        self.cur = self.con.cursor()

        self.height = cfg.height
        self.base_font_size = cfg.base_font_size
        self.n_table_rows = cfg.n_table_rows

        analysis_cfg_dict = db.read_misc("analysis_cfg", self.cur)
        self.analysis_cfg = AnalysisConfig.from_dict(analysis_cfg_dict)
        self.dataset_info = db.read_misc("dataset_info", self.cur)
        self.model_info = db.read_misc("model_info", self.cur)

        self.sae_ids = db.read_sae_ids(self.cur)
        self.sae_id = self.sae_ids[0]
        self.sae_data = db.read_sae_data(self.sae_ids[0], self.cur)

        self.table_ranking_option: "RankingOption" = {
            "kind": "feature_id",
            "descending": True,
        }
        self.table_min_act_rate = (
            cfg.default_min_act_rate
            if cfg.default_min_act_rate is not None
            else cfg.default_min_act_instances / self.dataset_info["n_sequences"]
            if cfg.default_min_act_instances is not None
            else 0
        )
        self.table_page_index = 0
        self.num_filtered_features = self.sae_data["n_alive_features"]
        self.max_table_page_index = (
            math.ceil(self.num_filtered_features / self.n_table_rows) - 1
        )
        self.table_features = db.rank_features(
            self.sae_id,
            self.cur,
            self.table_ranking_option,
            self.table_min_act_rate,
            self.table_page_index,
            self.n_table_rows,
            len(self.dataset_info["labels"]),
        )

        self.detail_feature = self.table_features[0]
        self.detail_feature_id = self.detail_feature["feature_id"]

        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae

        if (
            self.model is not None
            and self.tokenizer is not None
            and self.sae is not None
        ):
            self.can_inference = True
            self.model.to(cfg.device)  # type: ignore
        else:
            self.can_inference = False

        if not self.can_inference and (
            model is not None or tokenizer is not None or sae is not None
        ):
            raise ValueError(
                "model, tokenizer, and sae must all be set in order to inference"
            )

        self.inference_input: "InferenceInput" = {
            "feature_index": -1,
            "sequence": "",
        }
        self.inference_output: "FeatureTokenSequence" = {
            "feature_index": -1,
            "sequence_index": -1,
            "display_tokens": [],
            "max_token_index": -1,
            "label": -1,
            "pred_label": -1,
            "pred_probs": [],
            "extras": {},
        }

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
        self._update_table_features()

    @traitlets.observe("table_ranking_option")
    def table_ranking_option_change(self, _):
        """When the ranking option is changed, go back to the first page.
        Updating table_features will happen in the change handler for
        table_page_index. If we are already on the first page,
        then update table_features here."""

        if self.table_page_index == 0:
            self._update_table_features()
        else:
            self.table_page_index = 0

    @traitlets.observe("table_min_act_rate")
    def _on_table_min_act_rate(self, _):
        """When the table is filtered, go back to the first page.
        Updating table_features will happen in the change handler for
        table_page_index. If we are already on the first page,
        then update table_features here."""

        self.num_filtered_features = db.count_features(
            self.sae_id,
            self.cur,
            self.table_min_act_rate,
        )
        self.max_table_page_index = (
            math.ceil(self.num_filtered_features / self.n_table_rows) - 1
        )

        if self.table_page_index == 0:
            self._update_table_features()
        else:
            self.table_page_index = 0

    def _update_table_features(self):
        self.table_features = db.rank_features(
            self.sae_id,
            self.cur,
            self.table_ranking_option,
            self.table_min_act_rate,
            self.table_page_index,
            self.n_table_rows,
            len(self.dataset_info["labels"]),
        )

    @traitlets.observe("inference_input")
    def _on_inference_input_change(self, _):
        if self.inference_input["feature_index"] == -1:
            return

        if not self.can_inference:
            return

        if self.model is None or self.tokenizer is None or self.sae is None:
            return

        self.inference_output = inference(
            self.inference_input,
            self.model,
            self.tokenizer,
            self.sae,
            self.analysis_cfg,
        )
