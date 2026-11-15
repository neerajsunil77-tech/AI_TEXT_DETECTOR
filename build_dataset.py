
import csv
from pathlib import Path

DATASET_FILE = Path("training_data.csv")

CATEGORIES = ["english", "math", "algorithm"]
LABELS = ["human", "ai"]

FIELDS = ["text", "label", "category"]


def create_dataset():
    if not DATASET_FILE.exists():
        with open(DATASET_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()

    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != FIELDS:
            print("Your CSV columns are incorrect.")
            return

        rows = list(reader)

    print("\n=== English AI Detector Dataset Builder ===")
    print("1. Add a human-written example")
    print("2. Add an AI-generated example")
    print("3. Show dataset statistics")
    print("4. Exit")

    while True:
        choice = input("\nChoose an option: ").strip()

        if choice == "4":
            print("Dataset saved. Goodbye!")
            break

        elif choice == "3":
            print("\n=== Dataset Quality Report ===")

            for category in CATEGORIES:
                print(f"\n{category.upper()}")

                for label in LABELS:
                    count = sum(
                        1 for row in rows
                        if row["category"] == category
                        and row["label"] == label
                    )
                    print(f"{label.capitalize()}: {count}")

                    if count < 100:
                        print("  More verified examples needed.")

                human_count = sum(
                    1 for row in rows
                    if row["category"] == category
                    and row["label"] == "human"
                )
                ai_count = sum(
                    1 for row in rows
                    if row["category"] == category
                    and row["label"] == "ai"
                )

                if human_count >= 100 and ai_count >= 100:
                    print("  Initial example target reached!")
        elif choice in ["1", "2"]:
            label = "human" if choice == "1" else "ai"

            print("\nChoose category:")
            for number, category in enumerate(CATEGORIES, 1):
                print(f"{number}. {category}")

            category_choice = input("Category number: ").strip()

            if category_choice not in ["1", "2", "3"]:
                print("Invalid category.")
                continue

            category = CATEGORIES[int(category_choice) - 1]

            text = input("\nEnter the example text:\n").strip()

            if len(text) < 20:
                print("Please enter at least 20 characters.")
                continue

                        # Prevent duplicate examples
            if any(row["text"].strip().lower() == text.lower()
                   for row in rows):
                print("This example already exists!")
                continue

                        # Prevent duplicate examples
            if any(row["text"].strip().lower() == text.lower()
                   for row in rows):
                print("This example already exists!")
                continue

            rows.append({
                "text": text,
                "label": label,
                "category": category
            })

            with open(
                DATASET_FILE, "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            print(f"Saved as {label} / {category}.")

        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    create_dataset()
