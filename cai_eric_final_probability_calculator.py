import numpy as np
import pandas as pd
import json

def clean_text(text):
    text = text.lower()
    text = text.strip()
    for character in text:
        if character in r"""[]!.,"-!—@;':#$%^&*()+/?\\<>""":
            text = text.replace(character, " ")
    text = text.replace("\ufeff", " ")
    return text

def remove_stop_words(words, stop_words):
    words_copy = list(words)
    for word in words_copy:
        if word in stop_words:
            words.remove(word)
    return words

def count_words(words, is_spam, counted):
    for word in words:
        if word in counted:
            if is_spam == 1:
                counted[word][0] += 1
            else:
                counted[word][1] += 1
        else:
            if is_spam == 1:
                counted[word] = [1, 0]
            else:
                counted[word] = [0, 1]
    return counted

def make_percent_list(counted, num_spam, num_ham):
    for key, value in counted.items():
        converted_percent = (counted[key][0] + 1) / (num_spam + 2)
        counted[key][0] = converted_percent
        converted_percent = (counted[key][1] + 1) / (num_ham + 2)
        counted[key][1] = converted_percent
    return counted

counted = {}
num_lines = 0
num_spam = 0
num_ham = 0
is_spam = 0

training_file = 'Data/youtube_comments_train.csv'
comments = pd.read_csv(training_file)

with open('Data/StopWords.txt', 'r', encoding='unicode-escape') as f:
    stop_words = []
    line = f.readline()
    while line != "":
        stop_words.append(line.strip())
        line = f.readline()

num_spam = np.count_nonzero(comments['class'])
num_ham = len(comments) - num_spam

for comment_text, is_spam in zip(comments['content'], comments['class']):
    comment_text = clean_text(comment_text)
    comment_words = comment_text.split()
    comment_words = set(comment_words)
    comment_words = remove_stop_words(comment_words, stop_words)
    counted = count_words(comment_words, is_spam, counted)

counted = make_percent_list(counted, num_spam, num_ham)

with open('cai_eric_final_probs.txt', 'w', encoding='unicode-escape') as f:
    json.dump(counted, f)

with open('cai_eric_final_counts.txt', 'w') as f:
    counts = f"{num_ham} {num_spam}"
    f.write(counts)

print("Successfully calculated probabilities.")

sorted_counted = sorted(counted.items(), key=lambda x : x[1][0], reverse=True)

print("TOP 20 SPAM WORDS")
for i in range(20):
    print(f"{i + 1}. {sorted_counted[i][0]}")
    
sorted_counted = sorted(counted.items(), key=lambda x : x[1][1], reverse=True)

print("\nTOP 20 HAM WORDS")
for i in range(20):
    print(f"{i + 1}. {sorted_counted[i][0]}")