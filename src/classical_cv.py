from pathlib import Path
import cv2
import matplotlib.pyplot as plt


IMAGE_PATH = Path("data/raw/bloodcells_dataset/neutrophil/BNE_358773.jpg")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_image(image_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb, image_bgr


def create_opencv_pipeline(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median_blur = cv2.medianBlur(gray, 5)

    hist_equalized = cv2.equalizeHist(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    _, otsu_threshold = cv2.threshold(
        clahe_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    opening = cv2.morphologyEx(
        otsu_threshold,
        cv2.MORPH_OPEN,
        kernel,
    )

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel,
    )

    canny_edges = cv2.Canny(gaussian_blur, 50, 150)

    return {
        "Original": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        "Grayscale": gray,
        "Gaussian Blur": gaussian_blur,
        "Median Blur": median_blur,
        "Histogram Equalization": hist_equalized,
        "CLAHE": clahe_image,
        "Otsu Threshold": otsu_threshold,
        "Morphological Opening": opening,
        "Morphological Closing": closing,
        "Canny Edges": canny_edges,
    }


def save_pipeline_figure(processed_images: dict):
    plt.figure(figsize=(18, 10))

    for index, (title, image) in enumerate(processed_images.items()):
        plt.subplot(2, 5, index + 1)

        if title == "Original":
            plt.imshow(image)
        else:
            plt.imshow(image, cmap="gray")

        plt.title(title)
        plt.axis("off")

    plt.suptitle(
        "Classical Computer Vision Pipeline for Microscopy Image Analysis",
        fontsize=18,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "opencv_pipeline.png", dpi=300)
    plt.close()


def main():
    _, image_bgr = load_image(IMAGE_PATH)

    processed_images = create_opencv_pipeline(image_bgr)
    save_pipeline_figure(processed_images)

    print(f"Loaded image: {IMAGE_PATH}")
    print("Applied OpenCV preprocessing steps:")
    for step in processed_images.keys():
        print(f"- {step}")

    print("\nSaved figure: reports/figures/opencv_pipeline.png")


if __name__ == "__main__":
    main()