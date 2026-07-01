from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import build_model


MODEL_PATH = Path("models/resnet18_blood_cell_classifier.pth")

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_trained_model(device=None):
    if device is None:
        device = get_device()

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_names = checkpoint["class_names"]

    return model, class_names, device


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    image_tensor = transform(image).unsqueeze(0)

    return image, image_tensor


def predict_image(image_path):
    model, class_names, device = load_trained_model()
    original_image, image_tensor = preprocess_image(image_path)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    prediction = {
        "predicted_class": class_names[predicted_idx.item()],
        "confidence": confidence.item(),
        "probabilities": {
            class_names[i]: probabilities[i].item()
            for i in range(len(class_names))
        },
    }

    return prediction


if __name__ == "__main__":
    sample_image = "data/raw/bloodcells_dataset/neutrophil/BNE_358773.jpg"
    result = predict_image(sample_image)

    print("Prediction result")
    print("-" * 30)
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nClass probabilities:")
    for class_name, probability in result["probabilities"].items():
        print(f"{class_name}: {probability:.4f}")