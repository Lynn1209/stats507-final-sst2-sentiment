# src/baseline_tfidf.py

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

from eval_utils import plot_confusion_matrix
from data import get_label_names




def main():
    # 1. 加载数据
    dataset = load_dataset("glue", "sst2")
    train_texts = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]
    val_texts = dataset["validation"]["sentence"]
    val_labels = dataset["validation"]["label"]

    # 2. 构建 TF-IDF 特征
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    # 3. 训练逻辑回归模型
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)

    # 4. 在验证集上评估
    val_pred = clf.predict(X_val)
    acc = accuracy_score(val_labels, val_pred)
    f1 = f1_score(val_labels, val_pred)

    print("=== TF-IDF + Logistic Regression on SST-2 ===")
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation F1:       {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(val_labels, val_pred, digits=4))

    # 5. 画混淆矩阵
    label_map = get_label_names()
    labels = [label_map[0], label_map[1]]
    plot_confusion_matrix(
        y_true=val_labels,
        y_pred=val_pred,
        labels=labels,
        title="TF-IDF + Logistic Regression (validation)",
        save_path="reports/figures/confusion_matrix_tfidf_lr.png",
    )


if __name__ == "__main__":
    main()
