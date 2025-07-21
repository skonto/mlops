import os

import onnx
from onnx import shape_inference


def inspect_onnx_batch_dim(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = onnx.load(model_path)

    print(f"\n Inspecting ONNX model: {model_path}")
    print(f"  IR Version: {model.ir_version}")
    print(f"  Opset Version: {model.opset_import[0].version}")
    print(f"  Producer: {model.producer_name} {model.producer_version}")

    # Inferred shapes (optional but useful)
    model = shape_inference.infer_shapes(model)

    def shape_str(shape):
        dims = []
        for dim in shape.dim:
            if dim.dim_param:
                dims.append(dim.dim_param)  # e.g. "batch_size"
            elif dim.dim_value > 0:
                dims.append(str(dim.dim_value))
            else:
                dims.append("?")
        return "[" + ", ".join(dims) + "]"

    def check_dynamic(shape):
        return (
            shape.dim[0].dim_param != "" or shape.dim[0].dim_value == 0
        )

    print("\n Inputs:")
    for input_tensor in model.graph.input:
        shape = input_tensor.type.tensor_type.shape
        name = input_tensor.name
        print(f"  - {name}: {shape_str(shape)}")
        if not check_dynamic(shape):
            print(f"Batch dimension is static! Triton won't batch this.")

    print("\n📤 Outputs:")
    for output_tensor in model.graph.output:
        shape = output_tensor.type.tensor_type.shape
        name = output_tensor.name
        print(f"  - {name}: {shape_str(shape)}")
        if not check_dynamic(shape):
            print(f"Output batch dimension is static!")

    print("\nDone.\n")


# Example usage
if __name__ == "__main__":
    inspect_onnx_batch_dim("model.onnx")  # Adjust path as needed
