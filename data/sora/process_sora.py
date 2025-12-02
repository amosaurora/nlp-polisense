import csv
import json
from collections import defaultdict

def custom_round(value):
    return int(value + 0.5)

final_result = []
articles_data = defaultdict(list)

with open("Sora_LREC2020_biasedsentences.csv", mode="r", encoding="utf-8") as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    for row in csv_reader:
        if len(row) < 31:
            continue
        articles_data[row[3]].append(row)

for article_id, rows in articles_data.items():
    first_row = rows[0]
    all_scores = []
    for row in rows:
        scores = []
        for i in range(9, 31):
            if i < len(row) and row[i].strip().isdigit():
                scores.append(int(row[i]))
            else:
                scores.append(0)
        all_scores.append(scores)
    
    num_participants = len(all_scores)
    averages = []
    for i in range(22):
        total = sum(participant[i] for participant in all_scores)
        avg = total / num_participants
        averages.append(custom_round(avg))
    
    entry = {
        "id_event": first_row[0],
        "event": first_row[1],
        "date_event": first_row[2],
        "id_article": first_row[3],
        "source": first_row[4],
        "source_bias": first_row[5],
        "url": first_row[6],
        "ref": first_row[7],
        "ref_url": first_row[8],
        "article_bias": averages[0],
        "t": averages[1]
    }
    
    for i in range(20):
        entry[f"{i}"] = averages[2 + i]
    
    extra_fields = [
        "preknow", "reftitle", "reftext", "doctitle", "docbody",
        "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9",
        "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19"
    ]
    
    for i, field in enumerate(extra_fields):
        col = 31 + i
        value = first_row[col] if col < len(first_row) else ""
        if field.startswith("s") and field[1:].isdigit():
            sent_num = int(field[1:])
            prefix = f"[{sent_num}]:"
            if value.startswith(prefix):
                value = value[len(prefix):].strip()
        entry[field] = value
    
    final_result.append(entry)

with open("all_entries.json", "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=4)
