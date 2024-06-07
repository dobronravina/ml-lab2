import random
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def calculate_metrics(df, model_name, thresholds):
    metrics = {
        "threshold": [],
        "accuracy": [], "precision": [], "recall": [],
        "F1": [], "MCC": [], "BA": [], "J": [],
        "TPR": [], "FPR": []
    }

    for t in thresholds:
        TP = ((df["GT"] == 1) & (df[model_name] > t)).sum()
        FP = ((df["GT"] == 0) & (df[model_name] > t)).sum()
        TN = ((df["GT"] == 0) & (df[model_name] <= t)).sum()
        FN = ((df["GT"] == 1) & (df[model_name] <= t)).sum()

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        F1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        MCC = ((TP * TN) - (FP * FN)) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) if (TP + FP) * (
                    TP + FN) * (TN + FP) * (TN + FN) > 0 else 0
        BA = 0.5 * (TP / (TP + FN) + TN / (TN + FP)) if TP + FN > 0 and TN + FP > 0 else 0
        J = TP / (TP + FN) + TN / (TN + FP) - 1 if TP + FN > 0 and TN + FP > 0 else 0
        TPR = recall
        FPR = FP / (FP + TN) if FP + TN > 0 else 0

        metrics["threshold"].append(t)
        metrics["accuracy"].append(accuracy)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["F1"].append(F1)
        metrics["MCC"].append(MCC)
        metrics["BA"].append(BA)
        metrics["J"].append(J)
        metrics["TPR"].append(TPR)
        metrics["FPR"].append(FPR)

    return metrics


def plot_metrics(metrics, model_name):
    plt.figure(figsize=(10, 6))
    for metric in ["accuracy", "precision", "recall", "F1", "MCC", "BA", "J"]:
        plt.plot(metrics["threshold"], metrics[metric], label=metric)

    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title(f'Metrics for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_precision_recall_curve(df, model_name):
    precision, recall, _ = precision_recall_curve(df["GT"], df[model_name])
    auc_prc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f'PRC curve (area = {auc_prc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_roc_curve(df, model_name):
    fpr, tpr, _ = roc_curve(df["GT"], df[model_name])
    auc_roc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()


def modify_dataframe(df, percentage_to_delete):
    class_1_rows = df[df["GT"] == 1]
    rows_to_delete = random.sample(list(class_1_rows.index), int(len(class_1_rows) * (percentage_to_delete / 100)))
    return df.drop(rows_to_delete)


def main():
    col_names = ["GT", "Model_1", "Model_2"]
    df = pd.read_csv("KM-12-1.csv", names=col_names)

    # Convert columns to numeric, coercing errors
    df["GT"] = pd.to_numeric(df["GT"], errors='coerce')
    df["Model_1"] = pd.to_numeric(df["Model_1"], errors='coerce')
    df["Model_2"] = pd.to_numeric(df["Model_2"], errors='coerce')

    # Drop rows with NaN values
    df.dropna(inplace=True)

    print("Columns in the dataframe:", df.columns)

    if 'GT' not in df.columns:
        print("Error: 'GT' column not found in the dataframe.")
        return

    print(df["GT"].value_counts())

    thresholds = [i / 100 for i in range(100, -1, -10)]

    for model_name in ["Model_1", "Model_2"]:
        metrics = calculate_metrics(df, model_name, thresholds)
        plot_metrics(metrics, model_name)
        plot_precision_recall_curve(df, model_name)
        plot_roc_curve(df, model_name)

    df_modified = modify_dataframe(df, 50 + 10 * (8 % 4))
    print(df_modified["GT"].value_counts())


if __name__ == "__main__":
    main()



