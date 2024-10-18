import json
import math

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

def calculate_log_conditional_probability(b_given_a, b_given_not_a, a, not_a):
    a_given_b = 1 / (1 + math.e ** (math.log(b_given_not_a * not_a) - math.log(b_given_a * a)))
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
    subject_spam_log_probability = calculate_log_conditional_probability(
        spam_subject_probability,
        not_spam_subject_probability,
        spam_probability,
        not_spam_probability)
    return subject_spam_probability, subject_spam_log_probability

# Read in the dictionary and spam ham counts.
with open('cai_eric_P4_probs.txt', encoding='unicode-escape') as f:
    word_probabilities = json.load(f)
with open('cai_eric_P4_counts.txt') as f:
    count_string = f.readline()
    counts = count_string.split()
    counts = [int(count) for count in counts]

test_file = input("Enter the name of your test file: ")

actual_spam_ham = []
predicted_spam_ham = []
log_predicted_spam_ham = []

with open(test_file, 'r', encoding='unicode-escape') as f:
    
    for line in f:
        # Create a list of each email's actual category.
        actual_spam_ham.append(line[0])
    
        # Create a list of each email's predicted category.
        subject_spam_prob, subject_spam_log_prob = calculate_spam_probability(line, word_probabilities, counts)
        if subject_spam_prob > 0.50:
            predicted_spam_ham.append("1")
        else:
            predicted_spam_ham.append("0")

        # Create a list of each email's predicted category with the log method.
        if subject_spam_log_prob > 0.50:
            log_predicted_spam_ham.append("1")
        else:
            log_predicted_spam_ham.append("0")

# Evaluate the accuracy of the prediction.
tp = 0
fp = 0
tn = 0
fn = 0

for actual, prediction in zip(actual_spam_ham, predicted_spam_ham):
    if actual == '1':
        if prediction == '1':
            tp += 1
        else:
            fn += 1
    else:
        if prediction == '1':
            fp += 1
        else:
            tn += 1

print("\nMultiplication Method")
print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"TN: {tn}")
print(f"FN: {fn}")
accuracy = (tp + tn) / (tp + fp + tn + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")

# Evaluate the accuracy of the prediction with the log method.
log_tp = 0
log_fp = 0
log_tn = 0
log_fn = 0

for actual, prediction in zip(actual_spam_ham, log_predicted_spam_ham):
    if actual == '1':
        if prediction == '1':
            log_tp += 1
        else:
            log_fn += 1
    else:
        if prediction == '1':
            log_fp += 1
        else:
            log_tn += 1

print("\nLog Method")
print(f"TP: {log_tp}")
print(f"FP: {log_fp}")
print(f"TN: {log_tn}")
print(f"FN: {log_fn}")
log_accuracy = (log_tp + log_tn) / (log_tp + log_fp + log_tn + log_fn)
log_precision = log_tp / (log_tp + log_fp)
log_recall = log_tp / (log_tp + log_fn)
log_f1 = 2 * (precision * recall) / (precision + recall)
print(f"Accuracy: {log_accuracy}")
print(f"Precision: {log_precision}")
print(f"Recall: {log_recall}")
print(f"F1: {log_f1}")