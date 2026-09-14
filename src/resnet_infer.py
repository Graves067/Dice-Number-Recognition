"""
Inference wrapper for the trained die-face ResNet classifier.

Shaped to match Number_Detect.NumberDetection's interface — a `.read(crop)`
method returning (label, confidence) — so this can be swapped in for
NumberDetection in Main.py / Number_Detect.watch_folder with minimal
changes once there's a trained checkpoint. Until then, NumberDetection
(EasyOCR) keeps doing the job.

Usage:
    detector = ResnetNumberDetection()
    number, confidence = detector.read(crop)
"""
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from dice_dataset import default_transforms, CLASSES
from resnet_model import build_model

DEFAULT_CHECKPOINT = r"src\models\dice_resnet_best.pt"


class ResnetNumberDetection:
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT, device: str = None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_path}. Run train_resnet.py first."
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classes = checkpoint.get("classes", CLASSES)

        self.model = build_model(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.transform = default_transforms(train=False)

    def read(self, crop: np.ndarray):
        """
        Takes a cropped die face image (BGR NumPy array, same as
        NumberDetection.read expects) and returns (label, confidence).
        """
        # BGR (OpenCV) -> RGB (PIL/torchvision)
        rgb = crop[:, :, ::-1]
        img = Image.fromarray(rgb)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)

        label = self.classes[pred_idx.item()]
        return label, confidence.item()