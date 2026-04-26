"""Singleton loader for Qwen2.5-Omni-7B-GPTQ-Int4 on a local GPU.

The model and processor are loaded once on first call to infer() and kept in
memory for the lifetime of the process. A threading lock prevents concurrent
load attempts. Load failures are cached so every participant does not trigger
a redundant reload attempt.

Loading strategy: gptqmodel >= 4.0.0 is required because the checkpoint was
quantized with gptqmodel 4.0.0-dev and the earlier 2.0.0 release has an
incompatible dequantize_weight implementation (zeros/scales shape mismatch).
gptqmodel 4.2.5+ has Qwen2.5-Omni registered in MODEL_MAP natively, so no
custom patching is needed.

The speaker dictionary (spk_dict.pt) stored alongside the safetensor shards
must be loaded separately; gptqmodel's safetensor loader does not pick it up
automatically.

Required environment (separate conda env) — see requirements-local.txt.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch

from pmsv_synth.config import QWEN_LOCAL_MODEL_ID

_lock = threading.Lock()
_model: Any = None
_processor: Any = None
_load_error: BaseException | None = None  # cached so we fail fast after first failure


def _find_model_local_path(model_id: str) -> str | None:
    """Return the local HuggingFace cache directory for model_id, or None."""
    try:
        from huggingface_hub import hf_hub_download
        sentinel = hf_hub_download(model_id, "config.json", local_files_only=True)
        return os.path.dirname(sentinel)
    except Exception:
        return None


def _load_speaker_dict(model: Any, model_id: str) -> None:
    """Populate model.speaker_map from spk_dict.pt in the model directory.

    Qwen2.5-Omni stores speaker embeddings in a separate spk_dict.pt file
    alongside the safetensor shards.  The model's load_speakers() method
    must be called explicitly — it is not part of the safetensor loading.
    Without this the speaker_map is empty and generate() raises ValueError.
    """
    model_dir = _find_model_local_path(model_id)
    if model_dir is None:
        print(
            "[qwen-local] Warning: could not locate model cache dir — "
            "speaker_map will be empty.",
            flush=True,
        )
        return

    spk_path = os.path.join(model_dir, "spk_dict.pt")
    if os.path.exists(spk_path) and hasattr(model, "load_speakers"):
        model.load_speakers(spk_path)
        print(
            f"[qwen-local] Loaded {len(model.speaker_map)} speaker(s): "
            f"{list(model.speaker_map.keys())}",
            flush=True,
        )
    else:
        print(
            "[qwen-local] Warning: spk_dict.pt not found or load_speakers "
            "unavailable — speaker_map will be empty.",
            flush=True,
        )


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
            from gptqmodel import GPTQModel as _GPTQModel
        except ImportError as exc:
            _load_error = exc
            raise ImportError(
                "gptqmodel not found. Install it:\n"
                "  pip install gptqmodel>=4.0.0"
            ) from exc

        try:
            from transformers import Qwen2_5OmniProcessor
        except (ImportError, AttributeError) as exc:
            _load_error = exc
            raise ImportError(
                "Qwen2_5OmniProcessor not found. Install the preview fork:\n"
                "  pip install git+https://github.com/huggingface/"
                "transformers@v4.51.3-Qwen2.5-Omni-preview"
            ) from exc

        print(f"[qwen-local] Loading model {QWEN_LOCAL_MODEL_ID} …", flush=True)
        try:
            # gptqmodel >= 4.0.0 has qwen2_5_omni in MODEL_MAP natively
            # and correctly handles the desc_act=True, sym=True quantization
            # format used by the Qwen2.5-Omni-7B-GPTQ-Int4 checkpoint.
            gptq = _GPTQModel.from_quantized(
                QWEN_LOCAL_MODEL_ID,
                device="cuda:0",
            )
            _model = gptq.model
            _model.eval()

            # Load the speaker dictionary (spk_dict.pt) that the model card
            # stores separately from the safetensor shards.  Without this,
            # model.speaker_map is empty and generate() raises ValueError.
            _load_speaker_dict(_model, QWEN_LOCAL_MODEL_ID)

            _processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_LOCAL_MODEL_ID)
        except Exception as exc:
            _load_error = exc
            raise

        print("[qwen-local] Model loaded.", flush=True)
    return _model, _processor


def infer(
    system_prompt: str,
    user_text: str,
    video_path: Path,
    *,
    max_new_tokens: int = 512,
) -> str:
    """Run one video+text inference call and return the raw text response.

    Audio in video is always included — the task involves rating audio
    properties (strong_soundeffect, strong_soundtrack, etc.).
    """
    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:
        raise ImportError(
            f"Failed to import qwen_omni_utils ({exc}). "
            "Make sure qwen-omni-utils and torchvision are installed:\n"
            "  pip install qwen-omni-utils[decord]\n"
            "  pip install torchvision --index-url https://download.pytorch.org/whl/cu128"
        ) from exc

    model, processor = _load()

    # Cap frames and resolution to avoid OOM on long / high-res videos.
    #
    # Without limits the visual encoder produces ~2 880 tokens/frame at native
    # 1008×560 resolution; 32 frames × 2 880 = 92 160 tokens — well beyond the
    # 32 768 token context window. Downscaling to 336×196 (≈ 24× fewer pixels)
    # and 32 frames yields ~32 × 168 ≈ 5 376 visual tokens, which fits
    # comfortably alongside the text prompt and 512 generated tokens.
    conversation: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "nframes": 32,
                    "resized_height": 196,
                    "resized_width": 336,
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
        audios=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )

    # Detect device; gptqmodel wrappers may not expose .parameters() directly.
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_audio_in_video=True,
            return_audio=False,
            do_sample=False,
            # Explicitly set EOS so generation stops after the assistant turn.
            # model.config.eos_token_id is None for Qwen2.5-Omni; the thinker
            # and tokenizer both use 151645 (<|im_end|>) as the EOS token.
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # generate() returns a tensor when return_audio=False; guard against tuple
    output_ids = output[0] if isinstance(output, (tuple, list)) else output

    input_len = inputs["input_ids"].shape[1]
    output_text = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text
