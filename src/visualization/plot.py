import seaborn as sns

from matplotlib import pyplot as plt


def plot_loss_and_val_loss(train_feedback, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(train_feedback.history["loss"], label="Training Loss")
    plt.plot(train_feedback.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_loss_and_val_accuracy(train_feedback, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(train_feedback.history["accuracy"], label="Training accuracy")
    plt.plot(train_feedback.history["val_accuracy"], label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_confusion_matrix_heatmap(confusion_matrix, labels, title="Confusion heatmap"):
    _, (ax) = plt.subplots(1, 1, figsize=(15, 45))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        fmt="d",
        cmap=plt.cm.Blues,
        cbar=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
