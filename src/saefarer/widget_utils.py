import json
import os
from pathlib import Path
from typing import Any, Union

from saefarer.feature_data import FeatureData


def read_json(path: Union[str, os.PathLike]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_feature_data(
    feature_index: int, sae_data: Any, root_dir: Union[str, os.PathLike]
) -> FeatureData:
    return read_json(Path(root_dir) / sae_data["feature_index_to_path"][feature_index])


def convert_keys_to_ints(dictionary):
    """Convert string keys that are integers into integers."""
    new_dictionary = {}
    for key, value in dictionary.items():
        if isinstance(key, str) and key.isdigit():
            new_dictionary[int(key)] = value
        else:
            new_dictionary[key] = value
    return new_dictionary
