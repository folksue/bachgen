# Bach Chorale Transformer

This project trains a Transformer language model for polyphonic Bach chorales and generates new samples as MIDI and MusicXML.

## Representation Choice

We use a simple event-based symbolic representation tailored to 4-part chorales. Each time step contains a TIME token followed by four voice tokens (S, A, T, B). Each voice token is a MIDI pitch value (0-127). A special REST value 128 indicates silence.

Token layout:

- BOS, then repeated blocks of: TIME, note0, note1, note2, note3, then EOS
- All tokens are mapped to integer IDs for the Transformer LM

This keeps the model causal and easy to decode back into a 4-part score.

## Route A: From Scratch (this repo)

### 1) Prepare Data
Place CSV files with columns `note0,note1,note2,note3` under `data/train/` (and optionally `data/val/`).

Example:
```
/mnt/win_c/bachgen/data/train/chorale_000.csv
```

### 2) Train
```
python -m src.train --data-dir data/train --out-dir runs/from_scratch
```

### 3) Generate
```
python -m src.generate --checkpoint runs/from_scratch/model.pt --out-dir outputs
```

Outputs:
- `outputs/sample_000.mid`
- `outputs/sample_000.musicxml`

## Route B: Fine-Tune a Pretrained Model (two options)

### Option B1: Symbolic, GPT-2 as a pretrained LM baseline
Use the same tokenization from this repo, but initialize a `transformers` causal LM from a pretrained `gpt2` checkpoint, resize embeddings to the chorale vocab, and fine-tune on the chorale tokens. This is a fast, practical fine-tuning baseline when no music-specific checkpoints are available.

High-level steps:
1. Convert CSVs to token sequences (reuse `src/representation.py`).
2. Use Hugging Face `AutoModelForCausalLM.from_pretrained("gpt2")`.
3. Resize token embeddings to `vocab_size`.
4. Fine-tune with standard causal LM loss.

### Option B2: Music-specific pretrained model (recommended if you have compute)
For SOTA-style fine-tuning on symbolic music, you can use a music-specific model (e.g., Music Transformer or REMI-based checkpoints). If you have access to a checkpoint, plug in its tokenizer and finetune on chorales by mapping to its tokenization (usually REMI or CP). This can improve musical coherence over a general LM baseline.

If you want, I can add a full fine-tuning script for either option once you confirm the target checkpoint.

## Notes
- Export to MusicXML uses `music21`, so the result opens cleanly in MuseScore.
- You can adjust `step-duration` during generation to control rhythmic density.
