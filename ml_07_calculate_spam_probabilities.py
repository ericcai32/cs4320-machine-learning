import json

def clean_text(text):
    text = text.lower()
    text = text.strip()
    for letters in text:
        if letters in """[]!.,"-!—@;':#$%^&*()+/?""":
            text = text.replace(letters, " ")
    return text

def calculate_conditional_probability(b_given_a, b_given_not_a, a, not_a):
    a_given_b = (b_given_a * a) / ((b_given_a * a) + (b_given_not_a * not_a))
    return a_given_b

def calculate_spam_probability(subject, word_probabilities, spam_ham_counts):
    # Calculate P(sl|S), P(sl|~S), P(S) and P(~S)
    spam_subject_probability = 1
    not_spam_subject_probability = 1
    subject = clean_text(subject)
    subject_words = subject.split()
    subject_words = set(subject_words)
    for word, probabilities in word_probabilities.items():
        if word in subject_words:
            spam_subject_probability *= probabilities[0]
            not_spam_subject_probability *= probabilities[1]
        else:
            spam_subject_probability *= 1 - probabilities[0]
            not_spam_subject_probability *= 1 - probabilities[1]
    total_count = sum(spam_ham_counts)
    spam_probability = spam_ham_counts[0] / total_count
    not_spam_probability = spam_ham_counts[1] / total_count
    subject_spam_probability = calculate_conditional_probability(
        spam_subject_probability,
        not_spam_subject_probability,
        spam_probability,
        not_spam_probability)
    print(f"P(sl|S): {spam_subject_probability}")
    print(f"P(sl|~S): {not_spam_subject_probability}")
    print(f"P(S): {spam_probability}")
    print(f"P(~S): {not_spam_probability}")
    return subject_spam_probability

# Read in the dictionary and spam ham counts.
with open('eric_Dictionary.txt') as f:
    word_probabilities = json.load(f)
with open('HSCount.txt') as f:
    count_string = f.readline()
    counts = count_string.split()
    counts = [int(count) for count in counts]

subject = input("Please enter a subject line to test: ")
subject_spam_probability = calculate_spam_probability(subject, word_probabilities, counts)
print(f"P(S|sl): {subject_spam_probability}")