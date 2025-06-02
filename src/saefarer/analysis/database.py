import json
import sqlite3
from typing import TYPE_CHECKING, Any, Literal, Mapping

from saefarer.analysis.types import (
    FeatureData,
    LabelRankingOption,
    RankingOption,
    SAEData,
)

if TYPE_CHECKING:
    from pathlib import Path


def create_database(output_path: "Path") -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    con = sqlite3.connect(output_path.as_posix())
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE misc(
            key STRING PRIMARY KEY,
            value TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE sae(
            sae_id STRING PRIMARY KEY,
            n_total_features INTEGER,
            n_alive_features INTEGER,
            n_dead_features INTEGER,
            n_non_activating_features INTEGER,
            alive_feature_ids TEXT,
            token_act_rate_histogram TEXT,
            sequence_act_rate_histogram TEXT,
            feature_projection TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE feature(
            sae_id TEXT,
            feature_id INTEGER,
            max_act REAL,
            token_act_rate REAL,    
            token_acts_histogram TEXT,
            sequence_act_rate REAL,    
            sequence_acts_histogram TEXT,
            marginal_effects TEXT,
            cm TEXT,
            sequence_intervals TEXT,
            mean_pred_label_probs TEXT,
            PRIMARY KEY (sae_id, feature_id)
        )
    """)

    return con, cur


def insert_misc(key: str, value: Any, con: sqlite3.Connection, cur: sqlite3.Cursor):
    cur.execute(
        """
        INSERT INTO misc VALUES(
            :key,
            :value
        )
        """,
        {"key": key, "value": json.dumps(value)},
    )
    con.commit()


def insert_sae(data: SAEData, con: sqlite3.Connection, cur: sqlite3.Cursor):
    cur.execute(
        """
        INSERT INTO sae VALUES(
            :sae_id,
            :n_total_features,
            :n_alive_features,
            :n_dead_features,
            :n_non_activating_features,
            :alive_feature_ids,
            :token_act_rate_histogram,
            :sequence_act_rate_histogram,
            :feature_projection
        )
        """,
        convert_dict_for_db(data),
    )
    con.commit()


def insert_feature(data: FeatureData, con: sqlite3.Connection, cur: sqlite3.Cursor):
    cur.execute(
        """
        INSERT INTO feature VALUES(
            :sae_id,
            :feature_id,
            :max_act,
            :token_act_rate,
            :token_acts_histogram,
            :sequence_act_rate,
            :sequence_acts_histogram,
            :marginal_effects,
            :cm,
            :sequence_intervals,
            :mean_pred_label_probs
        )
        """,
        convert_dict_for_db(data),
    )
    con.commit()


def convert_dict_for_db(x: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: v if isinstance(v, (int, float, str)) else json.dumps(v)
        for k, v in x.items()
    }


def read_misc(key: str, cur: sqlite3.Cursor) -> Any:
    res = cur.execute(
        """
        SELECT * FROM misc WHERE key = ?
        """,
        (key,),
    )
    return json.loads(res.fetchone()[1])


def read_sae_ids(cur: sqlite3.Cursor) -> list[str]:
    res = cur.execute(
        """
        SELECT sae_id FROM sae
        """
    )
    rows = res.fetchall()
    return [row[0] for row in rows]


def read_sae_data(sae_id: str, cur: sqlite3.Cursor) -> SAEData:
    res = cur.execute(
        """
        SELECT * FROM sae WHERE sae_id = ?
        """,
        (sae_id,),
    )
    (
        sae_id,
        n_total_features,
        n_alive_features,
        n_dead_features,
        n_non_activating_features,
        alive_feature_ids,
        token_act_rate_histogram,
        sequence_act_rate_histogram,
        feature_projection,
    ) = res.fetchone()

    return SAEData(
        sae_id=sae_id,
        n_total_features=n_total_features,
        n_alive_features=n_alive_features,
        n_dead_features=n_dead_features,
        n_non_activating_features=n_non_activating_features,
        alive_feature_ids=json.loads(alive_feature_ids),
        token_act_rate_histogram=json.loads(token_act_rate_histogram),
        sequence_act_rate_histogram=json.loads(sequence_act_rate_histogram),
        feature_projection=json.loads(feature_projection),
    )


def row_to_feature_data(row: Any) -> FeatureData:
    (
        sae_id,
        feature_id,
        max_act,
        token_act_rate,
        token_acts_histogram,
        sequence_act_rate,
        sequence_acts_histogram,
        marginal_effects,
        cm,
        sequence_intervals,
        mean_pred_label_probs,
    ) = row

    return FeatureData(
        sae_id=sae_id,
        feature_id=feature_id,
        max_act=max_act,
        token_act_rate=token_act_rate,
        token_acts_histogram=json.loads(token_acts_histogram),
        sequence_act_rate=sequence_act_rate,
        sequence_acts_histogram=json.loads(sequence_acts_histogram),
        marginal_effects=json.loads(marginal_effects),
        cm=json.loads(cm),
        sequence_intervals=json.loads(sequence_intervals),
        mean_pred_label_probs=json.loads(mean_pred_label_probs),
    )


def read_feature_data(
    feature_id: int, sae_id: str, cur: sqlite3.Cursor
) -> FeatureData | None:
    res = cur.execute(
        """
        SELECT * FROM feature WHERE sae_id = ? AND feature_id = ? 
        """,
        (
            sae_id,
            feature_id,
        ),
    )

    row = res.fetchone()

    if row is None:
        return None

    return row_to_feature_data(row)


def rank_features(
    sae_id: str,
    cur: sqlite3.Cursor,
    ranking_option: RankingOption,
    min_act_rate: float,
    page_index: int,
    n_table_rows: int,
    n_labels: int,
) -> list[FeatureData]:
    if ranking_option["kind"] == "label":
        res = _rank_features_by_label(
            sae_id=sae_id,
            cur=cur,
            ranking_option=ranking_option,
            min_act_rate=min_act_rate,
            page_index=page_index,
            n_table_rows=n_table_rows,
            n_labels=n_labels,
        )
    elif (
        ranking_option["kind"] == "feature_id"
        or ranking_option["kind"] == "sequence_act_rate"
    ):
        res = _rank_features_by_col(
            sae_id=sae_id,
            cur=cur,
            col=ranking_option["kind"],
            is_descending=ranking_option["descending"],
            min_act_rate=min_act_rate,
            page_index=page_index,
            n_table_rows=n_table_rows,
        )
    else:
        raise ValueError("Unknown kind")

    rows = res.fetchall()

    return [row_to_feature_data(row) for row in rows]


def count_features(
    sae_id: str,
    cur: sqlite3.Cursor,
    min_act_rate: float,
) -> int:
    res = cur.execute(
        """
        SELECT COUNT(feature_id)
        FROM feature
        WHERE sae_id = :sae_id AND sequence_act_rate > :min_act_rate
        """,
        {
            "sae_id": sae_id,
            "min_act_rate": min_act_rate,
        },
    )

    row = res.fetchone()

    return row[0]


def _rank_features_by_col(
    sae_id: str,
    cur: sqlite3.Cursor,
    col: Literal["feature_id"] | Literal["sequence_act_rate"],
    is_descending: bool,
    min_act_rate: float,
    page_index: int,
    n_table_rows: int,
) -> sqlite3.Cursor:
    return cur.execute(
        f"""
        SELECT *
        FROM feature
        WHERE sae_id = :sae_id AND sequence_act_rate > :min_act_rate
        ORDER BY {col} {"DESC" if is_descending else "ASC"}
        LIMIT :limit
        OFFSET :offset
        """,
        {
            "sae_id": sae_id,
            "min_act_rate": min_act_rate,
            "limit": n_table_rows,
            "offset": n_table_rows * page_index,
        },
    )


def _rank_features_by_label(
    sae_id: str,
    cur: sqlite3.Cursor,
    ranking_option: LabelRankingOption,
    min_act_rate: float,
    page_index: int,
    n_table_rows: int,
    n_labels: int,
) -> sqlite3.Cursor:
    y_true = ranking_option["true_label"]
    y_pred = ranking_option["pred_label"]
    is_descending = ranking_option["descending"]

    if (y_true == "any" and y_pred == "any") or (
        y_true == "different" and y_pred == "different"
    ):
        return _rank_features_by_col(
            sae_id=sae_id,
            cur=cur,
            col="feature_id",
            is_descending=is_descending,
            min_act_rate=min_act_rate,
            page_index=page_index,
            n_table_rows=n_table_rows,
        )
    elif (y_true == "any" and y_pred == "different") or (
        y_true == "different" and y_pred == "any"
    ):
        return _rank_features_by_overall_error_pct(
            sae_id=sae_id,
            cur=cur,
            is_descending=is_descending,
            min_act_rate=min_act_rate,
            page_index=page_index,
            n_table_rows=n_table_rows,
        )
    elif y_pred == "any":
        return _rank_features_by_cm_value(
            sae_id=sae_id,
            cur=cur,
            key="label_pcts",
            label_index=int(y_true),
            is_descending=is_descending,
            min_act_rate=min_act_rate,
            page_index=page_index,
            n_table_rows=n_table_rows,
        )
    elif y_pred == "different":
        return _rank_features_by_cm_value(
            sae_id=sae_id,
            cur=cur,
            key="false_neg_pcts",
            label_index=int(y_true),
            is_descending=is_descending,
            min_act_rate=min_act_rate,
            page_index=page_index,
            n_table_rows=n_table_rows,
        )
    elif y_true == "any":
        return _rank_features_by_cm_value(
            sae_id=sae_id,
            cur=cur,
            key="pred_label_pcts",
            label_index=int(y_pred),
            is_descending=is_descending,
            min_act_rate=min_act_rate,
            page_index=page_index,
            n_table_rows=n_table_rows,
        )
    elif y_true == "different":
        return _rank_features_by_cm_value(
            sae_id=sae_id,
            cur=cur,
            key="false_pos_pcts",
            label_index=int(y_pred),
            is_descending=is_descending,
            min_act_rate=min_act_rate,
            page_index=page_index,
            n_table_rows=n_table_rows,
        )
    else:
        return _rank_features_by_cm_cell(
            sae_id=sae_id,
            cur=cur,
            true_label_index=int(y_true),
            pred_label_index=int(y_pred),
            is_descending=is_descending,
            min_act_rate=min_act_rate,
            page_index=page_index,
            n_table_rows=n_table_rows,
            n_labels=n_labels,
        )


def _rank_features_by_overall_error_pct(
    sae_id: str,
    cur: sqlite3.Cursor,
    is_descending: bool,
    min_act_rate: float,
    page_index: int,
    n_table_rows: int,
) -> sqlite3.Cursor:
    order = "DESC" if is_descending else "ASC"

    return cur.execute(
        f"""
        SELECT *
        FROM feature
        WHERE sae_id = :sae_id AND sequence_act_rate > :min_act_rate
        ORDER BY JSON_EXTRACT(cm, '$.error_pct') {order}
        LIMIT :limit
        OFFSET :offset
        """,
        {
            "sae_id": sae_id,
            "min_act_rate": min_act_rate,
            "limit": n_table_rows,
            "offset": n_table_rows * page_index,
        },
    )


def _rank_features_by_cm_value(
    sae_id: str,
    cur: sqlite3.Cursor,
    key: str,
    label_index: int,
    is_descending: bool,
    min_act_rate: float,
    page_index: int,
    n_table_rows: int,
) -> sqlite3.Cursor:
    order = "DESC" if is_descending else "ASC"

    return cur.execute(
        f"""
        SELECT *
        FROM feature
        WHERE sae_id = :sae_id AND sequence_act_rate > :min_act_rate
        ORDER BY JSON_EXTRACT(cm, '$.' || :key || '[' || :label_index || ']') {order}
        LIMIT :limit
        OFFSET :offset
        """,
        {
            "sae_id": sae_id,
            "min_act_rate": min_act_rate,
            "key": key,
            "label_index": label_index,
            "limit": n_table_rows,
            "offset": n_table_rows * page_index,
        },
    )


def _rank_features_by_cm_cell(
    sae_id: str,
    cur: sqlite3.Cursor,
    true_label_index: int,
    pred_label_index: int,
    is_descending: bool,
    min_act_rate: float,
    page_index: int,
    n_table_rows: int,
    n_labels: int,
) -> sqlite3.Cursor:
    index = true_label_index * n_labels + pred_label_index
    order = "DESC" if is_descending else "ASC"

    return cur.execute(
        f"""
        SELECT *
        FROM feature
        WHERE sae_id = :sae_id AND sequence_act_rate > :min_act_rate
        ORDER BY JSON_EXTRACT(cm, '$.cells[' || :index || '].pct') {order}
        LIMIT :limit
        OFFSET :offset
        """,
        {
            "sae_id": sae_id,
            "min_act_rate": min_act_rate,
            "index": index,
            "limit": n_table_rows,
            "offset": n_table_rows * page_index,
        },
    )
