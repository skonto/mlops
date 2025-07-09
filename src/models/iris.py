import torch.nn as nn
import torch

class IrisDL(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        median,
        iqr,
        activation="relu",
        dropout=0.0,
    ):
        super().__init__()

        # Store median and IQR as tensors (for ONNX compatibility)
        self.register_buffer("median", median )
        self.register_buffer("iqr",iqr)
        layers = []
        act_fn = nn.ReLU() if activation == "relu" else nn.GELU()

        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.median) / (self.iqr + 1e-6)
        return self.net(x)

    @classmethod
    def from_checkpoint(cls, path: str, config: dict, device: torch.device, compile_model: bool = False):
        model = cls(**config)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        if compile_model:
            model = torch.compile(model)

        return model