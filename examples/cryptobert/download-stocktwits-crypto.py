import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

sheet_to_df = pd.read_excel(
    "https://huggingface.co/datasets/ElKulako/stocktwits-crypto/resolve/main/st-data-full.xlsx",
    sheet_name=None,
)

df = pd.concat([sheet_to_df["stocktwits_1"], sheet_to_df["stocktwits_2"]]).dropna()

class_index_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
class_names = list(class_index_map.values())

df["text"] = df["text"].astype("string")
df["label"] = df["label"].map(class_index_map).astype("string")

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

dataset_dict.save_to_disk("stocktwits-crypto")
