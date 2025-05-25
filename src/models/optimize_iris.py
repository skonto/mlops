import optuna
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Full IrisDL implementation with tunable architecture
class IrisOpt(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation, dropout):
        super().__init__()
        act_fn = nn.ReLU() if activation == "relu" else nn.GELU()
        layers = []

        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def objective(trial):
    # Hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 5)
    hidden_dims = [trial.suggest_categorical(f"hidden_dim_{i}", [16, 32, 64, 128]) for i in range(n_layers)]
    activation = trial.suggest_categorical("activation", ["relu", "gelu"])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # Load and prepare data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Initialize model
    model = IrisOpt(input_dim=4, hidden_dims=hidden_dims, output_dim=3,
                   activation=activation, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(X_test_tensor)
        predicted_labels = torch.argmax(pred, dim=1)
        acc = accuracy_score(y_test_tensor.numpy(), predicted_labels.numpy())
    return acc

# Run the study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

trials = study.trials_dataframe()
plt.plot(trials["number"], trials["value"], marker="o", linestyle="-")
plt.title("Optuna Trial Accuracy")
plt.xlabel("Trial Number")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("opt_loss_plot.png")

best_trial = study.best_trial
best_trial_values = {
    "accuracy": best_trial.value,
    "params": best_trial.params
}

best_trial_values

