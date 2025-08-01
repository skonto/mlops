import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from models import IrisDL, TrainModelParams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(seed=100, train_size=0.8, num_epochs=1000, train_params=TrainModelParams()):
    set_seed(seed)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    median = np.median(X_train, axis=0)
    iqr = np.percentile(X_train, 75, axis=0) - np.percentile(X_train, 25, axis=0)

    model_params_dict = train_params.model.to_model_dict()
    train_params_dict = train_params.to_model_dict()
    model = IrisDL(**model_params_dict).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params_dict["lr"])

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    loss_history = []
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = loss_fn(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (epoch + 1) % 50 == 0:
            acc = (torch.argmax(y_pred, dim=1) == y_train_tensor).float().mean().item()
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Accuracy={acc*100:.2f}%")

    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.png")

    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred_test = model(X_test_tensor)
    preds = torch.argmax(y_pred_test, dim=1).cpu()

    print("Test Accuracy:", accuracy_score(preds, y_test))
    print("Confusion Matrix:\n", confusion_matrix(preds, y_test))
    return model, X, y


def train_and_export_model():
    model, X, y = train()
    model.eval()
    torch.save(model.state_dict(), "model.pt")

    model.eval()
    dummy_input = torch.randn(1, 4).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    return model, X, y


if __name__ == "__main__":
    model, X, y = train_and_export_model()
