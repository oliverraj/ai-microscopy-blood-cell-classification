from pathlib import Path
from PIL import Image
import pandas as pd


DATA_DIR = Path("data/raw/bloodcells_dataset")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


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


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    inventory_df = build_image_inventory()

    output_path = REPORT_DIR / "image_inventory.csv"
    inventory_df.to_csv(output_path, index=False)

    summarize_dataset(inventory_df)

    print(f"\nSaved inventory file to: {output_path}")


if __name__ == "__main__":
    main()