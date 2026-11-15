
import csv
from pathlib import Path

FILE = Path("training_data.csv")
FIELDS = ["text", "label", "category"]

CATEGORIES = ["english", "math", "algorithm"]
LABELS = ["human", "ai"]


def load_data():
    if not FILE.exists():
        return []

    with open(FILE, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_data(rows):
    with open(FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = load_data()
    known = {r["text"].strip().lower() for r in rows}

    print("\n=== BULK DATASET BUILDER ===")
    print("Enter multiple examples without restarting.")
    print("Type /done when you finish entering examples.\n")

    while True:
        print("\nCategory:")
        for i, category in enumerate(CATEGORIES, 1):
            print(f"{i}. {category}")

        choice = input("Choose category (1-3): ").strip()

        if choice == "/done":
            break

        if choice not in ("1", "2", "3"):
            print("Invalid choice.")
            continue

        category = CATEGORIES[int(choice) - 1]

        label_choice = input("Label (human/ai): ").strip().lower()

        if label_choice not in LABELS:
            print("Invalid label.")
            continue

        print("Enter your example. Type /done on a new line to finish:")

        lines = []

        while True:
            line = input()
            if line.strip() == "/done":
                break
            lines.append(line)

        text = "\n".join(lines).strip()

        if len(text) < 20:
            print("Example too short. Not saved.")
            continue

        if text.lower() in known:
            print("Duplicate example. Not saved.")
            continue

        rows.append({
            "text": text,
            "label": label_choice,
            "category": category
        })

        known.add(text.lower())
        save_data(rows)

        print(f"Saved: {label_choice} / {category}")

    print("\n=== DATASET SUMMARY ===")

    for category in CATEGORIES:
        print(f"\n{category.upper()}")

        for label in LABELS:
            count = sum(
                1 for r in rows
                if r["category"] == category
                and r["label"] == label
            )
            print(f"{label}: {count}")

    print("\nAll changes saved in training_data.csv")


if __name__ == "__main__":
    main()
