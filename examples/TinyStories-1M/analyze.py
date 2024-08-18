from saefarer.analyzing import analyze
from saefarer.config import AnalysisConfig
from saefarer.model import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_from_disk


def main():
    """Analyze the SAE"""

    cfg = AnalysisConfig(
        device="cuda",
        dataset_column="input_ids",
        model_batch_size_sequences=32,
        model_sequence_length=128,
        feature_batch_size=256,
        total_analysis_tokens=10_000_000,
        feature_indices=[],
        n_example_sequences=10,
        n_context_tokens=5,
    )

    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    sae = SAE.load("sae.pt", cfg.device)

    dataset = load_from_disk("TinyStories_tokenized_128")

    analyze(
        cfg=cfg,
        model=model,
        dataset=dataset,  # type: ignore
        sae=sae,
        decode_fn=tokenizer.batch_decode,  # type: ignore
        output_path="analysis.db",
    )


if __name__ == "__main__":
    main()
