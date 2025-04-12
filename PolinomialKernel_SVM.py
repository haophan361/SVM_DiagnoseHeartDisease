import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC

acc_svm = []
confusion_matrices = []
le = LabelEncoder()
df = pd.read_csv("heart.csv")
df = df.drop(columns=["RestingECG"])
df["Sex"] = le.fit_transform(df["Sex"])
df["ExerciseAngina"] = le.fit_transform(df["ExerciseAngina"])
df = pd.get_dummies(df, columns=['ChestPainType'], prefix='', prefix_sep='')
df = pd.get_dummies(df, columns=['ST_Slope'], prefix='', prefix_sep='')
target = "HeartDisease"
feature = df.columns.to_list()
feature.remove(target)
y = df[target].values

kf = model_selection.StratifiedKFold(n_splits=5)
avg_accuracy = 0

for fold, (train, value) in enumerate(kf.split(X=df, y=y)):
    X_train = df.loc[train, feature]
    heart_diseaseTrain = df.loc[train, target]
    X_valid = df.loc[value, feature]
    heart_diseaseTest = df.loc[value, target]

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_valid = ss.transform(X_valid)

    model_classification = SVC(kernel="rbf",gamma="scale",C=0.01, probability=True)
    model_classification.fit(X_train, heart_diseaseTrain)
    prob_predict = model_classification.predict_proba(X_valid)[:, 1]
    result_predict = model_classification.predict(X_valid)

    cm = confusion_matrix(heart_diseaseTest, result_predict)
    confusion_matrices.append(cm)

    acc = roc_auc_score(heart_diseaseTest, prob_predict)
    acc_svm.append(acc)
    avg_accuracy += acc

    report = classification_report(
        heart_diseaseTest, 
        result_predict, 
        target_names=['No Heart Disease', 'Heart Disease'], 
        output_dict=True
    )

    print(f"\nTập k thứ: {fold} :")
    for label in ['No Heart Disease', 'Heart Disease']:
        print(f"{label}: Precision: {report[label]['precision']:.2f}, Recall: {report[label]['recall']:.2f}, F1-Score: {report[label]['f1-score']:.2f}, Support: {int(report[label]['support'])}")
    print(f"Mức độ chính xác cho tập K thứ {fold}: {acc:.2f}")

avg_accuracy = avg_accuracy / 5
print("Tỉ lệ chính xác trung bình: " + str(avg_accuracy))

# Vẽ confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

labels_text = [['True Negative', 'False Positive'],
               ['False Negative', 'True Positive']]

for i, cm in enumerate(confusion_matrices):
    total = cm.sum()

    annotations = [[f"{labels_text[row][col]}\n{cm[row, col]}" 
                    for col in range(2)] for row in range(2)]
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', ax=axes[i],
                xticklabels=['No Heart Disease', 'Heart Disease'],
                yticklabels=['No Heart Disease', 'Heart Disease'])
    axes[i].set_title(f"Tập k thứ {i}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

for j in range(len(confusion_matrices), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
