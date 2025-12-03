import csv
import json

label_to_score = {
    "Expresses writer’s opinion": 4,
    "Somewhat factual but also opinionated": 3,
    "No agreement": 2,
    "Entirely factual": 1
}

final_result = []

with open("final_labels_SG1.csv", mode="r", encoding="utf-8") as file:
    csv_reader = csv.reader(file, delimiter=';')
    header = next(csv_reader)
    
    for row in csv_reader:
        if len(row) > 6:
            label_text = row[6]
            score = label_to_score.get(label_text)
            if score is not None:
                final_result.append({
                    "Text": row[0],
                    "Score": score
                })

with open("SG1_converted.json", "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

print(f"JSON saved with {len(final_result)} entries as SG1_converted.json")
