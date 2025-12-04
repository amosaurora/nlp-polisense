# import csv
# import json
# with open("final_labels_SG1.csv", mode="r", encoding="utf-8") as file:
#     csv_reader = csv.reader(file, delimiter=";")
#     header = next(csv_reader)

#     final = []

#     for row in csv_reader:
#         temp = {"Text":row[0],"Score":row[5]}
#         final.append(temp)
# with open("SG1_bias_dataset.json", "w", encoding="utf-8") as f:
#     json.dump(final, f, ensure_ascii=False, indent=4)

# import json

# # Load original JSON
# with open("Sora_bias_dataset.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# def convert_label(score):
#     if score == 1:
#         return "Non-biased"
#     elif score in (2,3, 4):
#         return "Biased"
#     else:
#         return "Unknown"

# # Create new list with converted labels
# output = []

# for item in data:
#     new_item = {
#         "Text": item["Text"],
#         "Score": convert_label(item["Score"])
#     }
#     output.append(new_item)

# # Save new JSON
# with open("output.json", "w", encoding="utf-8") as f:
#     json.dump(output, f, ensure_ascii=False, indent=4)

# import json

# # Load the first JSON file
# with open("SG1_bias_dataset.json", "r", encoding="utf-8") as f:
#     data1 = json.load(f)

# # Load the second JSON file
# with open("Sora_bias_dataset.json", "r", encoding="utf-8") as f:
#     data2 = json.load(f)

# # Combine the lists
# combined_data = data1 + data2

# # Save to a new JSON file
# with open("combined_dataset.json", "w", encoding="utf-8") as f:
#     json.dump(combined_data, f, ensure_ascii=False, indent=4)

# print("Saved: combined_dataset.json")

# import json

# # Load your JSON file
# with open("Sora_bias_dataset.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Mapping from text to integer
# score_map = {
#     "Non-biased": 0,
#     "Biased": 1,
#     "No agreement": 2
# }

# # Convert scores
# for item in data:
#     text_score = item.get("Score", "")
#     item["Score"] = score_map.get(text_score, -1)  # -1 for unknown

# # Save the reformatted JSON
# with open("output1.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)

# print("Saved: output_int_scores.json")

