import bz2
from io import StringIO

# from urllib.request import urlretrieve
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

# download the file
# urlretrieve("http://groups.di.unipi.it/~gulli/newsSpace.bz2", "ag-news.tsv.bz2")

# unzip it
with bz2.open("ag-news.tsv.bz2", mode="rt", encoding="latin1") as f:
    content = f.read()
    # replace escaped tab with space
    content = content.replace("\\\t", " ")
    # replace escaped new line with space
    content = content.replace("\\\n", " ")

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
    encoding="latin1",
    on_bad_lines=on_bad_lines,
)


class_index_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
class_names = list(class_index_map.values())

column_subset = ["title", "description", "category"]
df = df_original[df_original["category"].isin(class_names)][column_subset]

df.dropna(inplace=True)

df["text"] = df["title"] + " " + df["description"]

df.rename(columns={"category": "label"}, inplace=True)
df.drop(columns=["title", "description"], inplace=True)

features = Features({"text": Value("string"), "label": ClassLabel(names=class_names)})

dataset = Dataset.from_pandas(df, features=features, preserve_index=False)

# https://stackoverflow.com/a/76218276/5016634
train_testvalid = dataset.train_test_split(train_size=0.8, shuffle=True, seed=1)
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
