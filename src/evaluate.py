from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from dataset import create_dataloaders
from model import build_model


MODEL_PATH = Path("models/resnet18_blood_cell_classifier.pth")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_trained_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["class_names"]


def run_inference(model, validation_loader, class_names, device):
    true_labels = []
    predicted_labels = []
    confidences = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            confidence_values, predictions = torch.max(probabilities, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())
            confidences.extend(confidence_values.cpu().numpy())

    results_df = pd.DataFrame({
        "true_label_id": true_labels,
        "predicted_label_id": predicted_labels,
        "true_class": [class_names[i] for i in true_labels],
        "predicted_class": [class_names[i] for i in predicted_labels],
        "confidence": confidences,
    })

    return results_df


def save_metrics(results_df, class_names):
    y_true = results_df["true_label_id"]
    y_pred = results_df["predicted_label_id"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORT_DIR / "classification_report.csv")

    with open(REPORT_DIR / "evaluation_metrics.txt", "w") as file:
        file.write("ResNet18 Blood Cell Classifier Evaluation\n")
        file.write("=" * 45 + "\n")
        file.write(f"Accuracy:  {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall:    {recall:.4f}\n")
        file.write(f"Macro F1:  {f1:.4f}\n")

    print("\nEvaluation Metrics")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {f1:.4f}")

    return report_df


def plot_confusion_matrix(results_df, class_names):
    cm = confusion_matrix(
        results_df["true_label_id"],
        results_df["predicted_label_id"],
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - ResNet18 Blood Cell Classifier")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix_resnet18.png", dpi=300)
    plt.close()


def plot_class_performance(report_df):
    class_rows = report_df.iloc[:8].copy()

    plt.figure(figsize=(10, 5))
    class_rows["f1-score"].plot(kind="bar")
    plt.title("Class-wise F1 Score")
    plt.xlabel("Blood Cell Class")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_performance_f1.png", dpi=300)
    plt.close()


def main():
    device = get_device()
    print(f"Using device: {device}")

    _, validation_loader, _ = create_dataloaders()

    model, class_names = load_trained_model(device)

    results_df = run_inference(
        model=model,
        validation_loader=validation_loader,
        class_names=class_names,
        device=device,
    )

    results_df.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)

    report_df = save_metrics(results_df, class_names)

    plot_confusion_matrix(results_df, class_names)
    plot_class_performance(report_df)

    print("\nSaved evaluation outputs:")
    print("- reports/classification_report.csv")
    print("- reports/evaluation_metrics.txt")
    print("- reports/validation_predictions.csv")
    print("- reports/figures/confusion_matrix_resnet18.png")
    print("- reports/figures/class_performance_f1.png")


if __name__ == "__main__":
    main()