import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pandas as pd

from inputs_to_csv import inputs_to_csv

inputs_to_csv(["aa_accel_1.csv", "aa_accel_2.csv", "aa_accel_3.csv", "aa_accel_4.csv", "aa_accel_5.csv", "aa_accel_6.csv", "aa_accel_7.csv", "not_eating_aa_3.csv", "not_eating_aa_4.csv"],
              ["aa_is_eating_1.csv", "aa_is_eating_2.csv", "aa_is_eating_3.csv", "aa_is_eating_4.csv", "aa_is_eating_5.csv", "aa_is_eating_6.csv", "aa_is_eating_7.csv", "not_eating_aa_is_eating.csv", "not_eating_aa_is_eating.csv"], "train_aa_inputs_7.csv")

columns_to_use = [i for i in range(1, 38)]  # this line is for preventing an extra index column
test_day_inputs = pd.read_csv("train_aa_inputs_7.csv", usecols=columns_to_use)
test_day_inputs_array = test_day_inputs.to_numpy()
# np.savetxt("observations.txt", test_day_inputs_array, fmt="%s")

rng = np.random.default_rng()
rng.shuffle(test_day_inputs_array)

inputs = test_day_inputs_array[:, :-1]
labels = test_day_inputs_array[:, -1]

is_eating_percentage = np.sum(labels == 1) / len(labels) * 100
while is_eating_percentage < 40:
    indices_with_label_0 = np.where(labels == 0)[0]
    random_index = np.random.choice(indices_with_label_0)
    inputs = np.delete(inputs, random_index, axis=0)
    labels = np.delete(labels, random_index)
    is_eating_percentage = np.sum(labels == 1) / len(labels) * 100

print("number of segments where is_eating is 1: ", np.sum(labels == 1))
print("Percent of labels where is_eating is 1: ", is_eating_percentage)

training_inputs, test_inputs, \
    training_labels, test_labels = train_test_split(inputs, labels, test_size=0.2)

random_forest = RandomForestClassifier()
random_forest.fit(training_inputs, training_labels)

predictions = random_forest.predict(test_inputs)
overall_accuracy = accuracy_score(test_labels, predictions)
print(predictions)
print(test_labels)

conf_matrix = confusion_matrix(test_labels, predictions)
num_classes = len(conf_matrix)
per_class_accuracies = []
for i in range(num_classes):
    true_positive = conf_matrix[i, i]
    total_actual = sum(conf_matrix[i, :])
    per_class_accuracies.append(true_positive / total_actual)

print("Per-class Accuracies", per_class_accuracies)
print("Overall Accuracy: ", overall_accuracy)
ConfusionMatrixDisplay.from_estimator(random_forest, test_inputs, test_labels)
plt.show()
true_positives = conf_matrix[1, 1]
all_positives = conf_matrix[1, 1] + conf_matrix[0, 1]
false_negatives = conf_matrix[1, 0]
precision = true_positives / all_positives
recall = true_positives / (true_positives + false_negatives)
f_score = 2 * (precision * recall) / (precision + recall)
print("Precision: ", precision)
print("Recall: ", recall)
print("F-score: ", f_score)


# only one of dump and load should be uncommented

joblib.dump(random_forest, "random_forest.joblib")
# random_forest = joblib.load("random_forest.joblib")

inputs_to_csv(["aa_accel_8.csv"], ["aa_is_eating_8.csv"], "test_aa_inputs_8.csv")

test_day_inputs = pd.read_csv("test_aa_inputs_8.csv", usecols=columns_to_use)
test_day_inputs_array = test_day_inputs.to_numpy()
# test_day_inputs_array = np.loadtxt('test1.txt')



inputs = test_day_inputs_array[:, :-1]
labels = test_day_inputs_array[:, -1]

predictions = random_forest.predict(inputs)
overall_accuracy = accuracy_score(labels, predictions)
print(predictions)
print(labels)

conf_matrix = confusion_matrix(labels, predictions)
num_classes = len(conf_matrix)
per_class_accuracies = []
for i in range(num_classes):
    true_positive = conf_matrix[i, i]
    total_actual = sum(conf_matrix[i, :])
    per_class_accuracies.append(true_positive / total_actual)

print("Per-class Accuracies", per_class_accuracies)
print("Overall Accuracy: ", overall_accuracy)
ConfusionMatrixDisplay.from_estimator(random_forest, inputs, labels)
plt.show()
true_positives = conf_matrix[1, 1]
all_positives = conf_matrix[1, 1] + conf_matrix[0, 1]
false_negatives = conf_matrix[1, 0]
precision = true_positives / all_positives
recall = true_positives / (true_positives + false_negatives)
f_score = 2 * (precision * recall) / (precision + recall)
print("Precision: ", precision)
print("Recall: ", recall)
print("F-score: ", f_score)
