from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("data/raw/bloodcells_dataset")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_image_inventory() -> pd.DataFrame:
    records = []

    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        cell_class = class_dir.name

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode

                records.append(
                    {
                        "image_path": str(image_path),
                        "cell_class": cell_class,
                        "width": width,
                        "height": height,
                        "mode": mode,
                    }
                )

            except Exception as error:
                print(f"Could not read image: {image_path} | Error: {error}")

    return pd.DataFrame(records)


def summarize_dataset(df: pd.DataFrame) -> None:
    print("\nDataset Audit Summary")
    print("-" * 40)

    print(f"Total images: {len(df)}")
    print(f"Number of classes: {df['cell_class'].nunique()}")

    print("\nClass distribution:")
    print(df["cell_class"].value_counts())

    print("\nImage width summary:")
    print(df["width"].describe())

    print("\nImage height summary:")
    print(df["height"].describe())

    print("\nImage color modes:")
    print(df["mode"].value_counts())


def plot_class_distribution(df: pd.DataFrame) -> None:
    class_counts = df["cell_class"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    class_counts.plot(kind="bar")
    plt.title("Blood Cell Class Distribution")
    plt.xlabel("Cell Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=300)
    plt.close()


def plot_sample_images(df: pd.DataFrame) -> None:
    classes = sorted(df["cell_class"].unique())

    plt.figure(figsize=(14, 8))

    for index, cell_class in enumerate(classes):
        sample_path = df[df["cell_class"] == cell_class]["image_path"].iloc[0]

        with Image.open(sample_path) as img:
            plt.subplot(2, 4, index + 1)
            plt.imshow(img)
            plt.title(cell_class)
            plt.axis("off")

    plt.suptitle("Representative Blood Cell Images by Class", fontsize=16)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sample_images_by_class.png", dpi=300)
    plt.close()


def plot_image_size_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["width"], df["height"], alpha=0.3)
    plt.title("Image Size Distribution")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "image_size_distribution.png", dpi=300)
    plt.close()


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    inventory_df = build_image_inventory()

    output_path = REPORT_DIR / "image_inventory.csv"
    inventory_df.to_csv(output_path, index=False)

    summarize_dataset(inventory_df)

    plot_class_distribution(inventory_df)
    plot_sample_images(inventory_df)
    plot_image_size_distribution(inventory_df)

    print(f"\nSaved inventory file to: {output_path}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()