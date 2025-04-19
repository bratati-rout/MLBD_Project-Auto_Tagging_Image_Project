import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

input_file = r"E:\MLBD Project\data\captions.txt"
output_file = r"E:\MLBD Project\data\tag_list.txt"

all_words = set()

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",", 1)
        if len(parts) != 2:
            continue
        caption = parts[1].lower()
        words = caption.translate(str.maketrans("", "", string.punctuation)).split()
        filtered = [
            word for word in words
            if word not in stop_words and not word.isnumeric() and not any(char.isdigit() for char in word)
        ]
        all_words.update(filtered)

with open(output_file, "w", encoding="utf-8") as f:
    for word in sorted(all_words):
        f.write(word + "\n")

print(f"Done. Saved {len(all_words)} unique keywords to '{output_file}'")
