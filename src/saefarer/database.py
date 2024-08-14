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
            firing_rate_histogram BLOB,
            feature_projection BLOB
        )
    """)

    cur.execute("""
        CREATE TABLE feature(
            feature_id INTEGER,
            sae_id TEXT,
            firing_rate REAL,
            activations_histogram BLOB,
            sequences BLOB,
            PRIMARY KEY (feature_id, sae_id)
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
            :firing_rate_histogram,
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
            :feature_id,
            :sae_id,
            :firing_rate,
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
    row = res.fetchone()

    return SAEData(
        sae_id=row[0],
        num_dead_features=row[1],
        num_alive_features=row[2],
        alive_feature_ids=json.loads(row[3]),
        dead_feature_ids=json.loads(row[4]),
        firing_rate_histogram=json.loads(row[5]),
        feature_projection=json.loads(row[6]),
    )


def read_feature_data(feature_id: int, sae_id: str, cur: sqlite3.Cursor) -> FeatureData:
    res = cur.execute(
        """
        SELECT * FROM feature WHERE feature_id = ? AND sae_id = ?
        """,
        (
            feature_id,
            sae_id,
        ),
    )
    row = res.fetchone()

    return FeatureData(
        feature_id=row[0],
        sae_id=row[1],
        firing_rate=row[2],
        activations_histogram=json.loads(row[3]),
        sequences=json.loads(row[4]),
    )
