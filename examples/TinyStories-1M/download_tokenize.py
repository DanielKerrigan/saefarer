from saefarer.tokenize_and_concat import tokenize_dataset
from transformers import AutoTokenizer

from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

tokenized_dataset_128 = tokenize_dataset(
    dataset=dataset["train"],
    tokenizer=tokenizer,
    column_name="text",
    context_size=128,
    begin_batch_token_id=tokenizer.bos_token_id,
    begin_sequence_token_id=None,
    sequence_separator_token_id=tokenizer.bos_token_id,
    num_proc=8,
)
tokenized_dataset_128.save_to_disk("TinyStories_tokenized_128")
