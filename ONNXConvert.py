import torch
import torch.onnx

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import convert_graph_to_onnx as converter

from pathlib import Path

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-snli")

model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-snli")

def Convert_ONNX():
    model.eval()

    converter.convert(framework='pt',
                           model=model,
                           output=Path("onnx/bert-base-uncased-snli.onnx"),
                           opset=11,
                           tokenizer=tokenizer,
                           user_external_format=False,
                           pipeline_name="sentiment-analysis")
    print(" ")
    print("Model has been converted.")

if __name__ == "__main__":
    Convert_ONNX()