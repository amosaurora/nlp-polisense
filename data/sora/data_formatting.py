import json

with open('all_entries.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

formatted_score_data = []
formatted_leaning_data = []

for article in data:
    source_bias = article.get("source_bias", "")
    
    leaning = ""
    if "left" in source_bias.lower():
        leaning = "left"
    elif "right" in source_bias.lower():
        leaning = "right"
    elif "center" in source_bias.lower() or "least" in source_bias.lower():
        leaning = "center"
    else:
        leaning = "unknown"
    
    if "t" in article and "reftitle" in article:
        title = article["reftitle"]
        title_score = article["t"]
        
        if title and title.strip():
            formatted_score_data.append({
                "Text": title,
                "Score": title_score
            })
            formatted_leaning_data.append({
                "Text": title,
                "Leaning": leaning
            })
    
    for i in range(20):
        s_key = f"s{i}"
        score_key = str(i)
        
        if s_key in article and score_key in article:
            text = article[s_key]
            score = article[score_key]
            
            if text and text.strip():
                formatted_score_data.append({
                    "Text": text,
                    "Score": score
                })
                formatted_leaning_data.append({
                    "Text": text,
                    "Leaning": leaning
                })

with open('sentence_score.json', 'w', encoding='utf-8') as file:
    json.dump(formatted_score_data, file, ensure_ascii=False, indent=2)

with open('sentence_leaning.json', 'w', encoding='utf-8') as file:
    json.dump(formatted_leaning_data, file, ensure_ascii=False, indent=2)