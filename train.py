import torch
import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from activations_store import ActivationsStore
from config import Config
from sparse_autoencoder import SAE


def main():
    model = AutoModelForCausalLM.from_pretrained(
        "./saved_models/roneneldan/TinyStories-1M"
    )
    tokenizer = AutoTokenizer.from_pretrained("./saved_models/EleutherAI/gpt-neo-125M")

    dataset = load_from_disk("saved_datasets/tinystories")

    cfg = Config(
        d_sae=256,
        d_in=64,
        k=4,
        hidden_state_index=-2,
        normalize=True,
        lm_sequence_length=256,
        lm_batch_size_sequences=32,
        n_batches_in_store=20,
        sae_batch_size_tokens=4096,
        prepend_bos_token=True,
        device="cpu",
        dtype="float32",
    )

    sae = SAE(cfg)
    store = ActivationsStore(model, tokenizer, dataset["train"], cfg)

    num_tokens = 1e8
    batch_size = 2048

    num_batches = num_tokens // batch_size

    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4, betas=(0.9, 0.999))

    for i in tqdm.trange(num_batches):
        activations = store.next()
        loss, x_reconstruct, mid_acts, l2_loss, l1_loss = sae(activations)
        loss.backward()
        sae.set_decoder_norm_to_unit_norm()
        optimizer.step()
        optimizer.zero_grad()
        if i % 1000 == 0:
            print(loss.item())

    sae.save("sae.pt")


if __name__ == "__main__":
    main()
