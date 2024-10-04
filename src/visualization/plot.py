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
