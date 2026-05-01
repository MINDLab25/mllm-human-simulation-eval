from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

# --- Repo root (two levels up from this file: src/pmsv_synth/config.py) ---
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = ROOT_DIR / "data" / "samples"
OUTPUTS_DIR = ROOT_DIR / "outputs"

MSV_DF_PATH = DATA_DIR / "msv_df.csv"
MSV_BY_PARTICIPANT_PATH = DATA_DIR / "msv_df_by_participant.csv"

SAMPLES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# --- Env-sourced settings ---
# Optional when using --provider qwen3-local
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
VIDEO_DIR = Path(os.environ["VIDEO_DIR"])

SAMPLE_SIZE: int = int(os.environ.get("SAMPLE_SIZE", "3"))
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gemini-3-flash-preview")
MAX_CONCURRENT: int = int(os.environ.get("MAX_CONCURRENT", "10"))

# --- Local Qwen3-Omni-30B (--provider qwen3-local) ---
# Community GPTQ-4bit checkpoint; fits in 48 GB VRAM with disable_talker().
QWEN3_LOCAL_MODEL_ID: str = os.environ.get(
    "QWEN3_LOCAL_MODEL_ID", "Qwen/Qwen3-Omni-30B-A3B-Instruct"
)
