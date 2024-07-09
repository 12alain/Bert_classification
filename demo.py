import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
import pandas as pd
from tqdm import tqdm
# Importer les packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords') # Télécharger le package stopwords
nltk.download('wordnet')
from nltk.corpus import stopwords # Importer le package stopwords
import nltk
nltk.download('wordnet')
import gradio as gr
class CustomBertModel(nn.Module):
    def __init__(self, model_name_path="bert-base-uncased", n_classes=3):
        super(CustomBertModel, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(model_name_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x
    
model=CustomBertModel()
model.load_state_dict(torch.load("model.pth"))

def classifier_fn(text:str):
  labels={0:"Neutre",1:"negative", 2:"positive" }
  tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
  inputs=tokenizer(
      text,
      max_length=250,
      padding="max_length",
      truncation=True,
      return_tensors="pt"
  )
  output=model(inputs["input_ids"],inputs["attention_mask"])
  _,pred=output.max(1)
  
  return labels[pred.item()]

demo=gr.Interface(
    fn=classifier_fn,
    inputs=["text"],
    outputs=["text"],
)

demo.launch(share=True)