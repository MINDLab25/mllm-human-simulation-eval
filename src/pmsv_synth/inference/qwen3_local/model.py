"""Singleton loader for Qwen3-Omni-30B-A3B-Thinking (AWQ-4bit) on a local GPU.

The model and processor are loaded once on first call to infer() and kept in
memory for the lifetime of the process.  A threading lock prevents concurrent
load attempts.  Load failures are cached so every participant does not trigger
a redundant reload attempt.

Loading strategy
----------------
We load the community AWQ-4bit checkpoint directly via transformers.
Transformers detects the AWQ quantization config automatically from the model's
config.json — no BitsAndBytesConfig or GPTQConfig is needed.  The only extra
requirement is the autoawq library.

The Thinking variant is thinker-only (no Talker component), so:
  - No disable_talker() call is needed — it simply isn't there.
  - Saves ~10 GB vs the Instruct model at the same precision.
  - Memory estimate: ~10–15 GB weights (AWQ-4bit) + activations ≈ 25–30 GB total.

The Thinking model wraps its chain-of-thought inside <think>…</think> tags.
infer() strips these before returning so sync.py receives clean text suitable
for JSON parsing.

Required environment: pmsv-qwen3 conda env with autoawq installed.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import torch

from pmsv_synth.config import QWEN3_LOCAL_MODEL_ID

_lock = threading.Lock()
_model: Any = None
_processor: Any = None
_load_error: BaseException | None = None

# Matches the full <think>…</think> block the Thinking model emits before
# its actual answer.  re.DOTALL so . matches newlines inside the block.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> chain-of-thought prefix from model output."""
    return _THINK_RE.sub("", text).strip()


def _load() -> tuple[Any, Any]:
    global _model, _processor, _load_error
    with _lock:
        if _model is not None:
            return _model, _processor
        if _load_error is not None:
            raise RuntimeError(
                f"Model failed to load earlier ({_load_error}). "
                "Fix the error and restart the process."
            ) from _load_error

        try:
            from transformers import (
                Qwen3OmniMoeForConditionalGeneration,
                Qwen3OmniMoeProcessor,
            )
        except ImportError as exc:
            _load_error = exc
            raise ImportError(
                "transformers with Qwen3-Omni support not found. Install:\n"
                "  pip install git+https://github.com/huggingface/transformers\n"
                "  pip install bitsandbytes"
            ) from exc

        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            _load_error = exc
            raise ImportError("pip install bitsandbytes") from exc

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"[qwen3-local] Loading model {QWEN3_LOCAL_MODEL_ID} (bitsandbytes NF4) …", flush=True)
        try:
            _model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                QWEN3_LOCAL_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
            )
            _model.eval()

            # Thinking variant is thinker-only — no talker to disable.
            if hasattr(_model, "disable_talker"):
                _model.disable_talker()
                print("[qwen3-local] Talker disabled.", flush=True)

            _processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN3_LOCAL_MODEL_ID)
        except Exception as exc:
            _load_error = exc
            raise

        print("[qwen3-local] Model loaded.", flush=True)
    return _model, _processor


def infer(
    system_prompt: str,
    user_text: str,
    video_path: Path,
    *,
    max_new_tokens: int = 512,
) -> str:
    """Run one video+text inference call and return the raw text response.

    The Thinking model prepends a <think>…</think> block to every response.
    This is stripped before returning so callers receive only the JSON payload.
    max_new_tokens is higher than the Instruct model (1024 vs 512) to give the
    thinker enough budget for its reasoning chain.
    """
    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:
        raise ImportError(
            f"Failed to import qwen_omni_utils ({exc}). "
            "Install with: pip install qwen-omni-utils"
        ) from exc

    model, processor = _load()

    # Aggressively cap frames and resolution to fit within ~5 GB headroom.
    # The AWQ model occupies ~42 GB leaving little room for activations.
    # video_reader_backend="decord" avoids the torchvision.io.read_video issue.
    conversation: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "nframes": 16,
                    "resized_height": 112,
                    "resized_width": 168,
                    "video_reader_backend": "decord",
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_audio_in_video=True,
            return_audio=False,
            do_sample=False,
            thinker_return_dict_in_generate=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    if isinstance(output, (tuple, list)):
        thinker_out = output[0]
    else:
        thinker_out = output

    output_ids = thinker_out.sequences if hasattr(thinker_out, "sequences") else thinker_out

    input_len = inputs["input_ids"].shape[1]
    raw_text = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # Free activation memory before the next call.
    del inputs, output, output_ids
    torch.cuda.empty_cache()

    return _strip_thinking(raw_text)
