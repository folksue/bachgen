from pathlib import Path
from typing import List, Tuple

from music21 import note, stream


def merge_part_pitches(pitches: List[int]) -> List[Tuple[int, int]]:
    events: List[Tuple[int, int]] = []
    if not pitches:
        return events

    current_pitch = int(pitches[0])
    run_length = 1
    for pitch in pitches[1:]:
        pitch = int(pitch)
        if pitch == current_pitch:
            run_length += 1
        else:
            events.append((current_pitch, run_length))
            current_pitch = pitch
            run_length = 1
    events.append((current_pitch, run_length))
    return events


def merge_timesteps_to_events(timesteps: List[List[int]]) -> List[List[Tuple[int, int]]]:
    merged: List[List[Tuple[int, int]]] = []
    for idx in range(4):
        part_pitches = [int(step[idx]) for step in timesteps if len(step) > idx]
        merged.append(merge_part_pitches(part_pitches))
    return merged


def score_from_merged_events(
    merged_events: List[List[Tuple[int, int]]],
    step_duration: float = 0.25,
) -> stream.Score:
    score = stream.Score()
    part_names = ["Soprano", "Alto", "Tenor", "Bass"]
    parts = [stream.Part(id=name) for name in part_names]
    for part, name in zip(parts, part_names):
        part.partName = name
        score.append(part)

    for idx, events in enumerate(merged_events[:4]):
        for pitch, run_length in events:
            duration = step_duration * int(run_length)
            if pitch >= 128:
                event = note.Rest(quarterLength=duration)
            else:
                event = note.Note(int(pitch), quarterLength=duration)
            parts[idx].append(event)

    return score


def score_from_timesteps(timesteps: List[List[int]], step_duration: float = 0.25) -> stream.Score:
    merged_events = merge_timesteps_to_events(timesteps)
    return score_from_merged_events(merged_events, step_duration=step_duration)


def write_outputs(score: stream.Score, out_dir: Path, basename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_path = out_dir / f"{basename}.mid"
    xml_path = out_dir / f"{basename}.musicxml"
    score.write("midi", fp=str(midi_path))
    score.write("musicxml", fp=str(xml_path))
