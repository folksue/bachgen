from pathlib import Path
from typing import List

from music21 import note, stream


def score_from_timesteps(timesteps: List[List[int]], step_duration: float = 0.25) -> stream.Score:
    score = stream.Score()
    part_names = ["Soprano", "Alto", "Tenor", "Bass"]
    parts = [stream.Part(id=name) for name in part_names]
    for part, name in zip(parts, part_names):
        part.partName = name
        score.append(part)

    for step in timesteps:
        for idx, pitch in enumerate(step[:4]):
            if pitch >= 128:
                event = note.Rest(quarterLength=step_duration)
            else:
                event = note.Note(int(pitch), quarterLength=step_duration)
            parts[idx].append(event)

    return score


def write_outputs(score: stream.Score, out_dir: Path, basename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_path = out_dir / f"{basename}.mid"
    xml_path = out_dir / f"{basename}.musicxml"
    score.write("midi", fp=str(midi_path))
    score.write("musicxml", fp=str(xml_path))
