from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from inference import load_trained_model, preprocess_image


SAMPLE_IMAGE = Path("data/raw/bloodcells_dataset/neutrophil/BNE_358773.jpg")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()

        output = self.model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

        score = output[0, predicted_idx]
        score.backward()

        if self.gradients is None:
            raise RuntimeError("Gradients were not captured. Check target layer.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, predicted_idx.item(), confidence.item()


def overlay_heatmap(original_image, cam):
    original_resized = original_image.resize((224, 224))
    original_array = np.array(original_resized)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(0.6 * original_array + 0.4 * heatmap)

    return original_array, heatmap, overlay


def main():
    # Force CPU for Grad-CAM to avoid MPS hook issues
    device = torch.device("cpu")
    model, class_names, _ = load_trained_model(device=device)

    # Enable gradients for Grad-CAM visualization
    for parameter in model.parameters():
        parameter.requires_grad = True

    model.eval()

    original_image, input_tensor = preprocess_image(SAMPLE_IMAGE)
    input_tensor = input_tensor.to(device)

    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    cam, predicted_idx, confidence = gradcam.generate(input_tensor)
    original_array, heatmap, overlay = overlay_heatmap(original_image, cam)

    predicted_class = class_names[predicted_idx]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_array)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "gradcam_example.png", dpi=300)
    plt.close()

    print("Grad-CAM completed.")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("Saved figure: reports/figures/gradcam_example.png")


if __name__ == "__main__":
    main()