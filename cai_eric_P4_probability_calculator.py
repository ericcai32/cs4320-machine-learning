import json

def clean_text(text):
    text = text.lower()
    text = text.strip()
    for letters in text:
        if letters in """[]!.,"-!—@;':#$%^&*()+/?""":
            text = text.replace(letters, " ")
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

training_file = input("Enter the name to your training file: ")

with open('Data/StopWords.txt', 'r', encoding='unicode-escape') as f:
    stop_words = []
    line = f.readline()
    while line != "":
        stop_words.append(line.strip())
        line = f.readline()

with open(training_file, 'r', encoding='unicode-escape') as fin:
    line = fin.readline()
    while line != "":
        # Count the number of spam and ham emails.
        if line[0] == "1":
            is_spam = 1
            num_spam += 1
        else:
            is_spam = 0
            num_ham += 1
        
        # Create a set of cleaned subject lines.
        subject = clean_text(line[2:])
        subject = subject.split()
        subject = set(subject)
        subject = remove_stop_words(subject, stop_words)
        
        counted = count_words(subject, is_spam, counted)
        
        num_lines += 1
        
        line = fin.readline()

counted = make_percent_list(counted, num_spam, num_ham)

with open('cai_eric_P4_probs.txt', 'w', encoding='unicode-escape') as f:
    json.dump(counted, f)

with open('cai_eric_P4_counts.txt', 'w') as f:
    counts = f"{num_ham} {num_spam}"
    f.write(counts)

print("Successfully calculated probabilities.")