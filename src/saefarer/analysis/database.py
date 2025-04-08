import json
import sqlite3
from typing import TYPE_CHECKING, Any, Mapping

from saefarer.analysis.types import FeatureData, RankingOption, SAEData

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


def query_features(
    sae_id: str,
    cur: sqlite3.Cursor,
    ranking_option: RankingOption,
    page_index: int,
    n_table_rows: int,
) -> list[FeatureData]:
    res = cur.execute(
        """
        SELECT *
        FROM feature
        WHERE sae_id = ?
        ORDER BY feature_id DESC
        LIMIT 10
        """,
        (sae_id,),
    )

    rows = res.fetchall()

    return [row_to_feature_data(row) for row in rows]
