# %% imports
import bz2
from io import StringIO
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

# %% download the file
download_path = Path("ag-news.tsv.bz2")

if download_path.exists():
    print(f"{download_path} already exists, skipping download")
else:
    print("Downloading file")
    urlretrieve("http://groups.di.unipi.it/~gulli/newsSpace.bz2", download_path)

# %% unzip and do some minor cleaning
print("Reading file")
with bz2.open("ag-news.tsv.bz2", mode="rb") as f:
    content_bytes = f.read()
    content = content_bytes.decode("ascii", "replace")

    replacements = [
        ("\r", ""),  # remove \r
        ("\\\t", " "),  # replace escaped tab with space
        ("\n\\\n", "\\"),  # remove single backslash on own line
        ("\n...", "..."),  # remove new line before ...
        ("\\\n", "\\"),  # replace escaped new line with \
        ("�", ""),  # remove unknown
    ]

    for old, new in replacements:
        content = content.replace(old, new)


# %% parse the file

print("Parsing file")

columns = [
    "source",
    "url",
    "title",
    "image",
    "category",
    "description",
    "rank",
    "pubdate",
    "video",
]


def on_bad_lines(bad_line: list[str]) -> list[str] | None:
    print("bad line")

    print(f"{len(bad_line)=}")

    for i, x in enumerate(bad_line):
        print(i, x)

    print("\n")

    return None


df_original = pd.read_csv(
    StringIO(content),
    sep="\t",
    names=columns,
    engine="python",
    on_bad_lines=on_bad_lines,
    quoting=3,
)

print(f"{df_original.shape=}")
print(f"{df_original['category'].unique()=}")

# %% process the dataset

print("Processing dataframe")

class_names = ["World", "Sports", "Business", "Sci/Tech"]
column_subset = ["title", "description", "category"]


def row_filter(x):
    title_not_url = ~x["title"].str.contains("http")
    desc_not_url = ~x["description"].str.contains("http")
    desc_set = x["description"].str.strip() != "\\N"
    return title_not_url & desc_not_url & desc_set


df = (
    df_original[df_original["category"].isin(class_names)]
    .dropna(subset=column_subset)
    .drop_duplicates(subset=column_subset)
    .loc[row_filter]
    .assign(text=lambda df: df["title"] + " " + df["description"])
    .drop(columns=["url", "image", "description", "rank", "video"])
    .rename(columns={"category": "label"})
    .fillna("NA")
)

print(f"{df.shape=}")

# %% create hugging face dataset
# https://stackoverflow.com/a/76218276/5016634

print("Creating hugging face dataset")

features = Features(
    {
        "text": Value("string"),
        "source": Value("string"),
        "title": Value("string"),
        "pubdate": Value("string"),
        "label": ClassLabel(names=class_names),
    }
)
dataset = Dataset.from_pandas(df, features=features, preserve_index=False)

train_testvalid = dataset.train_test_split(train_size=0.9, shuffle=True, seed=1)
test_valid = train_testvalid["test"].train_test_split(
    train_size=0.5, shuffle=True, seed=2
)

dataset_dict = DatasetDict(
    {
        "train": train_testvalid["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"],
    }
)

dataset_dict.save_to_disk("ag-news")
