import numpy as np
import onnxruntime as ort
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix

from models import IrisDL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(model_path="model.pt"):
    iris = load_iris()
    X, y = iris.data, iris.target
    median = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)

    model = IrisDL(
        input_dim=4,
        hidden_dims=[16, 128, 32, 64],
        output_dim=3,
        median=torch.tensor(median, dtype=torch.float32).to(device),
        iqr=torch.tensor(iqr, dtype=torch.float32).to(device),
        activation="relu",
        dropout=0.1956
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, X, y


def test_onnx_export_consistency():
    model, X, y = load_trained_model("model.pt")

    x_numpy = np.array([X[0]], dtype=np.float32)
    x_tensor = torch.from_numpy(x_numpy).to(device)

    # Run inference in PyTorch
    with torch.no_grad():
        torch_output = model(x_tensor).to(device)
        
    use_cuda = torch.cuda.is_available()
    providers = ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession("model.onnx", providers=providers)
    onnx_output = ort_session.run(None, {ort_session.get_inputs()[0].name: x_numpy})[0]

    # Assert the predictions are numerically close
    assert np.allclose(torch_output.cpu().numpy(), onnx_output, rtol=1e-03, atol=1e-05), "ONNX output does not match PyTorch"

    # Optional: check predicted class
    torch_pred = torch.argmax(torch_output, dim=1).item()
    onnx_pred = np.argmax(onnx_output, axis=1)[0]
    assert torch_pred == onnx_pred, "ONNX and PyTorch predicted different classes"

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    full_preds = torch.argmax(model(X_tensor), dim=1).to(device)

    print("Full Accuracy:", accuracy_score(full_preds.cpu().numpy(), y))
    print("Full Confusion Matrix:\n", confusion_matrix(full_preds.cpu().numpy(), y))
