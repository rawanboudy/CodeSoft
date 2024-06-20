import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv('IRIS.csv')
print(iris.head())

label_encoder = preprocessing.LabelEncoder()
iris['species'] = label_encoder.fit_transform(iris['species'])

print(iris['species'].unique())


X = iris.drop(['species'], axis=1)
y = iris['species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


target_values = sorted(set(y))
y_list_train = [(y_train == i).astype(int) for i in target_values]
y_list_test = [(y_test == i).astype(int) for i in target_values]

model_lists = []
for i in range(len(target_values)):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_list_train[i])
    model_lists.append(model)

preds_prob_test = np.array([model.predict_proba(X_test)[:, 1] for model in model_lists]).T
predicted_classes = np.argmax(preds_prob_test, axis=1)

conf_matrix = confusion_matrix(y_test, predicted_classes)
accuracy = accuracy_score(y_test, predicted_classes)
precision = precision_score(y_test, predicted_classes, average='macro')
recall = recall_score(y_test, predicted_classes, average='macro')
f1 = f1_score(y_test, predicted_classes, average='macro')

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


Z = np.array([6.2, 3.5, 5.4, 2.3]).reshape(1, -1)

for j in range(len(model_lists)):
    preds = model_lists[j].predict(Z)
    print(f'Class {j} prediction: {preds[0]}')
for j in range(len(model_lists)):
    preds_prob = model_lists[j].predict_proba(Z)
    print(f'Class {j} probability: {preds_prob[0, 1]}')

predicted_class_Z = np.argmax([model.predict_proba(Z)[:, 1] for model in model_lists])
print(f'The predicted class for the sample {Z} is: {predicted_class_Z}')
