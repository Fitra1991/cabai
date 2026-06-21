import torch
from models.model import MultimodalDenseNet

def load_model():

    model = MultimodalDenseNet(
        num_classes=4
    )

    model.load_state_dict(
        torch.load(
            "models/fixmatch_mm_best.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model