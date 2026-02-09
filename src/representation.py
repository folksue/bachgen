from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class ChoraleTokenizer:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    time_id: int = 3
    rest_value: int = 128

    def __post_init__(self) -> None:
        self.note_offset = 4
        self.vocab_size = self.note_offset + (self.rest_value + 1)

    def note_to_id(self, value: Optional[int]) -> int:
        if value is None:
            value = self.rest_value
        if value < 0:
            value = self.rest_value
        if value > self.rest_value:
            value = self.rest_value
        return self.note_offset + int(value)

    def id_to_note(self, token_id: int) -> int:
        value = token_id - self.note_offset
        if value < 0:
            return self.rest_value
        if value > self.rest_value:
            return self.rest_value
        return value

    def encode_timesteps(self, timesteps: Iterable[Iterable[int]]) -> List[int]:
        tokens: List[int] = [self.bos_id]
        for step in timesteps:
            tokens.append(self.time_id)
            for note in step:
                tokens.append(self.note_to_id(note))
        tokens.append(self.eos_id)
        return tokens

    def decode_timesteps(self, tokens: Iterable[int]) -> List[List[int]]:
        timesteps: List[List[int]] = []
        buffer: List[int] = []
        reading = False
        for token in tokens:
            if token == self.eos_id:
                break
            if token == self.bos_id:
                continue
            if token == self.time_id:
                if buffer:
                    timesteps.append(buffer[:4])
                buffer = []
                reading = True
                continue
            if not reading:
                continue
            buffer.append(self.id_to_note(token))
            if len(buffer) == 4:
                timesteps.append(buffer)
                buffer = []
        if buffer:
            timesteps.append(buffer[:4])
        return timesteps
