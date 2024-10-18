import json

def clean_text(text):
    text = text.lower()
    text = text.strip()
    for letters in text:
        if letters in """[]!.,"-!—@;':#$%^&*()+/?""":
            text = text.replace(letters, " ")
    return text

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

file = 'Data/ClassExampleInput.txt'

counted = {}
num_lines = 0
num_spam = 0
num_ham = 0
is_spam = 0

with open(file) as f:
    line = f.readline()
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
        
        counted = count_words(subject, is_spam, counted)
        
        num_lines += 1
        
        line = f.readline()

print(counted)
print()
counted = make_percent_list(counted, num_spam, num_ham)
print(counted)

with open('eric_Dictionary.txt', 'w') as f:
    json.dump(counted, f)

with open('HSCount.txt', 'w') as f:
    counts = f"{num_spam} {num_ham}"
    f.write(counts)