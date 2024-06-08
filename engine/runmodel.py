from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from max import engine

HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest"
hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)

# Converting model to TorchScript
model_path = Path("roberta.torchscript")
batch = 1
seqlen = 128
inputs = {
    "input_ids": torch.zeros((batch, seqlen), dtype=torch.int64),
    "attention_mask": torch.zeros((batch, seqlen), dtype=torch.int64),
}
with torch.no_grad():
    traced_model = torch.jit.trace(
        hf_model, example_kwarg_inputs=dict(inputs), strict=False
    )

torch.jit.save(traced_model, model_path)

# We use the same `inputs` that we used above to trace the model
input_spec_list = [
    engine.TorchInputSpec(shape=tensor.size(), dtype=engine.DType.int64)
    for tensor in inputs.values()
]

session = engine.InferenceSession()
model = session.load(model_path, input_specs=input_spec_list)

for tensor in model.input_metadata:
    print(f'name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')

INPUT="There are many exciting developments in the field of AI Infrastructure!"

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
inputs = tokenizer(INPUT, return_tensors="pt", padding='max_length', truncation=True, max_length=seqlen)
print(inputs)

outputs = model.execute(**inputs)
print(outputs)

# Extract class prediction from output
predicted_class_id = outputs["result0"]["logits"].argmax(axis=-1)[0]
classification = hf_model.config.id2label[predicted_class_id]

print(f"The sentiment is: {classification}")