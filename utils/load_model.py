from collections import OrderedDict
from pathlib import Path
import os

# Batasi thread agar loading model PyTorch tidak lambat/hang di Streamlit Cloud.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from models.model import MultimodalDenseNet


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "fixmatch_mm_best.pth"


def _extract_state_dict(checkpoint):
    """Ambil state_dict dari berbagai format checkpoint."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

    return checkpoint


def _clean_state_dict(state_dict):
    """Hapus prefix umum dari checkpoint, misalnya module. dari DataParallel."""
    cleaned = OrderedDict()

    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value

    return cleaned


def load_model(model_path=None, num_classes=4, device="cpu"):
    """
    Load model multimodal DenseNet.

    Default device dibuat CPU karena Streamlit Cloud umumnya tidak menyediakan GPU.
    """
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch hanya mengizinkan set_num_interop_threads sebelum parallel work dimulai.
        pass

    path = Path(model_path) if model_path is not None else MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {path}. "
            "Pastikan fixmatch_mm_best.pth ada di folder models/."
        )

    device = torch.device(device)
    model = MultimodalDenseNet(num_classes=num_classes)

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    state_dict = _extract_state_dict(checkpoint)
    state_dict = _clean_state_dict(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model
