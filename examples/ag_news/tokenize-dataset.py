from datasets import load_from_disk
from transformers import AutoTokenizer

dataset = load_from_disk("ag-news")

model_name = "Kyle1668/ag-news-19200-bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenization(example):
    return tokenizer(
        example["text"], padding="max_length", max_length=128, truncation=True
    )


tokenized_dataset = dataset.map(
    tokenization,
    batched=True,
    num_proc=8,
)
tokenized_dataset.set_format(type="torch")


tokenized_dataset.save_to_disk("ag-news_tokenized")
