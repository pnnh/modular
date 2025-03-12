from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import numpy as np
from max import engine

# The Hugging Face model name and exported file name
HF_MODEL_NAME = "microsoft/resnet-50"
MODEL_PATH = "resnet50.onnx"

def main():
    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    image = dataset["test"]["image"][0]
    # 保存本地以方便查看
    image.save("cat.png")

    image_processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    inputs = image_processor(image, return_tensors="np")

    print("Keys:", inputs.keys())
    print("Shape:", inputs['pixel_values'].shape)

    session = engine.InferenceSession()
    model = session.load(MODEL_PATH)
    outputs = model.execute_legacy(**inputs)

    print("Output shape:", outputs['output'].shape)

    predicted_label = np.argmax(outputs["output"], axis=-1)[0]
    hf_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_NAME)
    predicted_class = hf_model.config.id2label[predicted_label]
    print(f"Prediction: {predicted_class}")

if __name__ == "__main__":
    main()