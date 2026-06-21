import torch
import torch.nn.functional as F

class_names = [
    "Bercak",
    "Keriting",
    "Kuning",
    "Sehat"
]

def predict(rgb_tensor, th_tensor, model):

    model.eval()

    with torch.no_grad():
        outputs = model(rgb_tensor, th_tensor)

        probs = F.softmax(outputs, dim=1)  # shape: [1, num_classes]

        confidence, predicted = torch.max(probs, 1)

    # ubah ke list agar bisa dipakai di streamlit
    probs_list = probs[0].tolist()

    return class_names[predicted.item()], confidence.item(), probs_list