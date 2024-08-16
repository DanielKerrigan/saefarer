import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from saefarer.types import FeatureData, SAEData


def create_database(output_path: Path) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    con = sqlite3.connect(output_path.as_posix())
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE sae(
            sae_id STRING PRIMARY KEY,
            num_alive_features INTEGER,
            num_dead_features INTEGER,
            alive_feature_ids BLOB,
            dead_feature_ids BLOB,
            activation_rate_histogram BLOB,
            dimensionality_histogram BLOB,
            cumsum_percent_l1_norm_range BLOB,
            feature_projection BLOB
        )
    """)

    cur.execute("""
        CREATE TABLE feature(
            sae_id TEXT,
            feature_id INTEGER,
            activation_rate REAL,    
            max_activation REAL,
            n_neurons_majority_l1_norm INTEGER,
            cumsum_percent_l1_norm BLOB,
            activations_histogram BLOB,
            sequences BLOB,
            PRIMARY KEY (sae_id, feature_id)
        )
    """)

    return con, cur


def insert_sae(data: SAEData, con: sqlite3.Connection, cur: sqlite3.Cursor):
    cur.execute(
        """
        INSERT INTO sae VALUES(
            :sae_id,
            :num_alive_features,
            :num_dead_features,
            :alive_feature_ids,
            :dead_feature_ids,
            :activation_rate_histogram,
            :dimensionality_histogram,
            :cumsum_percent_l1_norm_range,
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
            :activation_rate,
            :max_activation,
            :n_neurons_majority_l1_norm,
            :cumsum_percent_l1_norm,
            :activations_histogram,
            :sequences
        )
        """,
        convert_dict_for_db(data),
    )
    con.commit()


def convert_dict_for_db(x: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        k: v if isinstance(v, (int, float, str)) else json.dumps(v)
        for k, v in x.items()
    }


def read_sae_ids(cur: sqlite3.Cursor) -> List[str]:
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
        num_alive_features,
        num_dead_features,
        alive_feature_ids,
        dead_feature_ids,
        activation_rate_histogram,
        dimensionality_histogram,
        cumsum_percent_l1_norm_range,
        feature_projection,
    ) = res.fetchone()

    return SAEData(
        sae_id=sae_id,
        num_dead_features=num_dead_features,
        num_alive_features=num_alive_features,
        alive_feature_ids=json.loads(alive_feature_ids),
        dead_feature_ids=json.loads(dead_feature_ids),
        activation_rate_histogram=json.loads(activation_rate_histogram),
        dimensionality_histogram=json.loads(dimensionality_histogram),
        cumsum_percent_l1_norm_range=json.loads(cumsum_percent_l1_norm_range),
        feature_projection=json.loads(feature_projection),
    )


def read_feature_data(feature_id: int, sae_id: str, cur: sqlite3.Cursor) -> FeatureData:
    res = cur.execute(
        """
        SELECT * FROM feature WHERE sae_id = ? AND feature_id = ? 
        """,
        (
            sae_id,
            feature_id,
        ),
    )

    (
        sae_id,
        feature_id,
        activation_rate,
        max_activation,
        n_neurons_majority_l1_norm,
        cumsum_percent_l1_norm,
        activations_histogram,
        sequences,
    ) = res.fetchone()

    return FeatureData(
        sae_id=sae_id,
        feature_id=feature_id,
        activation_rate=activation_rate,
        max_activation=max_activation,
        n_neurons_majority_l1_norm=n_neurons_majority_l1_norm,
        cumsum_percent_l1_norm=json.loads(cumsum_percent_l1_norm),
        activations_histogram=json.loads(activations_histogram),
        sequences=json.loads(sequences),
    )
