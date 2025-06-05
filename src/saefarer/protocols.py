from typing import Protocol


class TokenizerProtocol(Protocol):
    all_special_ids: list[int]

    def decode(self, token_ids: list[int]) -> str: ...
