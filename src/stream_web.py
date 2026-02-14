import json
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Allow running as a script (python src/stream_web.py) by falling back to absolute imports
try:
    from .generate import sample_next
    from .midi_utils import score_from_merged_events, write_outputs
    from .model import ChoraleTransformerLM, ModelConfig
    from .representation import ChoraleTokenizer
except ImportError:
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.generate import sample_next
    from src.midi_utils import score_from_merged_events, write_outputs
    from src.model import ChoraleTransformerLM, ModelConfig
    from src.representation import ChoraleTokenizer


OUT_DIR = Path("outputs/stream_live")
BASENAME_CURRENT = "stream_current"
BASENAME_FINAL = "stream_final"
INDEX_HTML_PATH = Path("web/index.html")


class StartRequest(BaseModel):
    checkpoint: str = "runs/from_scratch/model_epoch_20.pt"
    max_steps: int = 512
    temperature: float = 1.0
    top_k: int = 32
    step_duration: float = 0.25
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    token_delay_ms: int = 0
    flush_every_steps: int = 4


@dataclass
class StreamStatus:
    running: bool = False
    done: bool = False
    error: Optional[str] = None
    revision: int = 0
    generated_tokens: int = 0
    generated_steps: int = 0
    checkpoint: str = ""
    out_dir: str = str(OUT_DIR)
    midi_url: Optional[str] = None
    musicxml_url: Optional[str] = None
    updated_at: float = 0.0


class TokenStepDecoder:
    def __init__(self, tokenizer: ChoraleTokenizer) -> None:
        self.tokenizer = tokenizer
        self.reading = False
        self.buffer: List[int] = []
        self.ended = False

    def consume(self, token_id: int) -> Optional[List[int]]:
        if self.ended:
            return None
        if token_id == self.tokenizer.bos_id:
            return None
        if token_id == self.tokenizer.eos_id:
            self.ended = True
            return None
        if token_id == self.tokenizer.time_id:
            self.buffer = []
            self.reading = True
            return None
        if not self.reading:
            return None

        self.buffer.append(self.tokenizer.id_to_note(token_id))
        if len(self.buffer) == 4:
            step = self.buffer
            self.buffer = []
            return step
        return None


class IncrementalMerger:
    def __init__(self) -> None:
        self.last_pitch: List[Optional[int]] = [None, None, None, None]
        self.run_length: List[int] = [0, 0, 0, 0]
        self.events: List[List[Tuple[int, int]]] = [[], [], [], []]

    def add_step(self, step: List[int]) -> None:
        for i in range(4):
            pitch = int(step[i])
            if self.last_pitch[i] is None:
                self.last_pitch[i] = pitch
                self.run_length[i] = 1
                continue
            if pitch == self.last_pitch[i]:
                self.run_length[i] += 1
            else:
                last = self.last_pitch[i]
                assert last is not None
                self.events[i].append((int(last), int(self.run_length[i])))
                self.last_pitch[i] = pitch
                self.run_length[i] = 1

    def snapshot_events(self, include_open_runs: bool = True) -> List[List[Tuple[int, int]]]:
        snap = [list(ev) for ev in self.events]
        if include_open_runs:
            for i in range(4):
                if self.last_pitch[i] is not None and self.run_length[i] > 0:
                    last = self.last_pitch[i]
                    assert last is not None
                    snap[i].append((int(last), int(self.run_length[i])))
        return snap

    def finalize(self) -> None:
        for i in range(4):
            if self.last_pitch[i] is not None and self.run_length[i] > 0:
                last = self.last_pitch[i]
                assert last is not None
                self.events[i].append((int(last), int(self.run_length[i])))
            self.last_pitch[i] = None
            self.run_length[i] = 0


_status_lock = threading.Lock()
_status = StreamStatus()
_stop_flag = threading.Event()
_worker_thread: Optional[threading.Thread] = None


def _load_tokenizer(checkpoint_path: Path) -> ChoraleTokenizer:
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    if tokenizer_path.exists():
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        return ChoraleTokenizer(
            pad_id=tok_cfg["pad_id"],
            bos_id=tok_cfg["bos_id"],
            eos_id=tok_cfg["eos_id"],
            time_id=tok_cfg["time_id"],
            rest_value=tok_cfg["rest_value"],
        )
    return ChoraleTokenizer()


def _load_model_config(checkpoint_path: Path, tokenizer: ChoraleTokenizer) -> ModelConfig:
    config_path = checkpoint_path.parent / "model_config.json"
    if not config_path.exists():
        return ModelConfig(vocab_size=tokenizer.vocab_size)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    allowed_keys = set(ModelConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in cfg.items() if k in allowed_keys}
    if "vocab_size" not in filtered:
        filtered["vocab_size"] = tokenizer.vocab_size
    return ModelConfig(**filtered)


def _write_snapshot(merger: IncrementalMerger, step_duration: float, basename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    score = score_from_merged_events(merger.snapshot_events(include_open_runs=True), step_duration=step_duration)
    write_outputs(score, OUT_DIR, basename)


def _generation_worker(req: StartRequest) -> None:
    global _status
    checkpoint_path = Path(req.checkpoint)

    try:
        tokenizer = _load_tokenizer(checkpoint_path)
        config = _load_model_config(checkpoint_path, tokenizer)
        model = ChoraleTransformerLM(config).to(req.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=req.device))
        model.eval()

        decoder = TokenStepDecoder(tokenizer)
        merger = IncrementalMerger()

        tokens: List[int] = [tokenizer.bos_id]

        with _status_lock:
            _status.running = True
            _status.done = False
            _status.error = None
            _status.revision = 0
            _status.generated_tokens = 1
            _status.generated_steps = 0
            _status.checkpoint = str(checkpoint_path)
            _status.midi_url = None
            _status.musicxml_url = None
            _status.updated_at = time.time()

        _stop_flag.clear()

        max_tokens = int(req.max_steps)
        delay_s = max(0.0, req.token_delay_ms / 1000.0)
        flush_every = max(1, int(req.flush_every_steps))

        with torch.no_grad():
            for _ in range(max_tokens):
                if _stop_flag.is_set():
                    break
                # IMPORTANT: learned positional embeddings are limited to config.max_len.
                # Keep generation stable by feeding only the last `max_len` tokens.
                if len(tokens) > int(config.max_len):
                    context = tokens[-int(config.max_len) :]
                else:
                    context = tokens
                input_ids = torch.tensor(context, dtype=torch.long, device=req.device).unsqueeze(0)
                logits = model(input_ids)[0, -1]
                next_id = sample_next(logits, req.temperature, req.top_k)
                tokens.append(int(next_id))

                with _status_lock:
                    _status.generated_tokens += 1

                step = decoder.consume(int(next_id))
                if step is not None:
                    merger.add_step(step)
                    do_flush = False
                    with _status_lock:
                        _status.generated_steps += 1
                        do_flush = (_status.generated_steps % flush_every) == 0
                    if do_flush:
                        _write_snapshot(merger, req.step_duration, BASENAME_CURRENT)
                        with _status_lock:
                            _status.revision += 1
                            _status.updated_at = time.time()
                            _status.midi_url = f"/files/{BASENAME_CURRENT}.mid?rev={_status.revision}"
                            _status.musicxml_url = f"/files/{BASENAME_CURRENT}.musicxml?rev={_status.revision}"

                if next_id == tokenizer.eos_id:
                    break
                if delay_s > 0:
                    time.sleep(delay_s)

        merger.finalize()
        _write_snapshot(merger, req.step_duration, BASENAME_FINAL)
        with _status_lock:
            _status.revision += 1
            _status.updated_at = time.time()
            _status.midi_url = f"/files/{BASENAME_FINAL}.mid?rev={_status.revision}"
            _status.musicxml_url = f"/files/{BASENAME_FINAL}.musicxml?rev={_status.revision}"
            _status.done = True

    except Exception:
        err = traceback.format_exc()
        with _status_lock:
            _status.error = err
            _status.done = True
    finally:
        with _status_lock:
            _status.running = False


app = FastAPI(title="BachGen Stream")

# static file mount
OUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(OUT_DIR)), name="files")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Missing web/index.html</h1>")


@app.get("/api/status")
def api_status() -> Dict:
    with _status_lock:
        return asdict(_status)


@app.post("/api/start")
def api_start(req: StartRequest) -> Dict:
    global _worker_thread
    with _status_lock:
        if _status.running:
            raise HTTPException(status_code=409, detail="generation already running")

    _worker_thread = threading.Thread(target=_generation_worker, args=(req,), daemon=True)
    _worker_thread.start()
    return {"ok": True}


@app.post("/api/stop")
def api_stop() -> Dict:
    _stop_flag.set()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.stream_web:app", host="0.0.0.0", port=8000, reload=False)
