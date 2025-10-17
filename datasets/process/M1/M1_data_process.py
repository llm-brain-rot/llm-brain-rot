from datasets import load_dataset
import json 

def is_english_basic(s):
    try:
        s.encode(encoding='ascii')
        return True
    except UnicodeEncodeError:
        return False
    
def is_mostly_ascii(s, threshold=0.9):
    ascii_count = sum(1 for c in s if ord(c) < 128)
    return ascii_count / max(len(s), 1) >= threshold

dataset = load_dataset("enryu43/twitter100m_tweets", split="train")

filtered = dataset.filter(
    lambda x:  x["likes"] > 500 and is_english_basic(x["tweet"]), num_proc=16
)

print(f"Filtered dataset size: {len(filtered)}")

sampled = filtered.shuffle(seed=42)

data_list = sampled.to_list()
# Save to a JSON file
with open("./twitter_100m_en.json", "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)


from transformers import AutoTokenizer


    
dataset = dataset.shuffle(seed=42)

high_qual = []
low_qual = []
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

for tweet in dataset["train"]:
    text= tweet["tweet"]
    token_len = len(tokenizer.encode(text, add_special_tokens=True))
    if is_english_basic(text):
        if token_len > 100 and len(high_qual) <= 15000:
            data_dict = {"text": text}
            high_qual.append(data_dict)
        if token_len <= 30 and len(low_qual) <= 80000:
            data_dict = {"text": text}
            low_qual.append(data_dict)
    if len(high_qual) > 15000 and len(low_qual) > 80000:
        break        

with open('high_quality_tweets_1m_len.json', 'w') as f:
    json.dump(high_qual, f, indent=4)

with open('low_quality_tweets_1m_len.json', 'w') as f:
    json.dump(low_qual, f, indent=4)