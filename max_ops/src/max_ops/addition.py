import numpy as np
from max import engine
from max.dtype import DType
from max.graph import Graph, TensorType, ops

def add_tensors(a: np.ndarray, b: np.ndarray) -> dict[str, any]:
    # 1. Build the graph
    input_type = TensorType(dtype=DType.float32, shape=(1,))
    with Graph(
        "simple_add_graph", input_types=(input_type, input_type)
    ) as graph:
        lhs, rhs = graph.inputs
        out = ops.add(lhs, rhs)
        graph.output(out)
    print("final graph:", graph)
    
    session = engine.InferenceSession()
    model = session.load(graph)
    for tensor in model.input_metadata:
        print(
            f"name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}"
        )
    ret = model.execute_legacy(input0=a, input1=b)
    print("result:", ret["output0"])
    return ret

if __name__ == "__main__":
    input0 = np.array([1.0], dtype=np.float32)
    input1 = np.array([1.0], dtype=np.float32)
    add_tensors(input0, input1)