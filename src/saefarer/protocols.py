from typing import Protocol, TypedDict


class EncodingOutput(TypedDict):
    token_ids: list[int]
    attention_mask: list[int]


class TokenizerProtocol(Protocol):
    pad_token_id: int

    def encode(
        self,
        text: str,
        max_length: int,
    ) -> EncodingOutput: ...

    def decode(self, token_ids: list[int]) -> str: ...
