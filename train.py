


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
# import sckeuder
from torch.optim import lr_scheduler


class LoadDataset(Dataset):

    def __init__(self, csv_file,device,model_name_path="bert-base-uncased", max_length=124):
        self.device=device
        self.df = pd.read_csv(csv_file)
        self.labels = self.df.sentiment.unique()
        labels_dict = {l: indx for indx, l in enumerate(self.labels)}

        self.df["sentiment"] = self.df["sentiment"].map(labels_dict)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        review_text = str(self.df.iloc[index]['text'])
        label_review = self.df.iloc[index]['sentiment']

        inputs = self.tokenizer(
            review_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = torch.tensor(label_review)
        return {
            "input_ids": inputs["input_ids"].squeeze(0).to(self.device),  # Remove batch dimension
            "attention_mask": inputs["attention_mask"].squeeze(0).to(self.device),
            "labels": labels.to(self.device)
        }

class CustomBertModel(nn.Module):
    def __init__(self, model_name_path="bert-base-uncased", n_classes=3):
        super(CustomBertModel, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(model_name_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x

def training_step(model, data_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]

        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader.dataset)

def evaluation(model, test_dataloader, loss_fn):
    model.eval()
    correct_predictions = 0
    losses = []

    for data in tqdm(test_dataloader, total=len(test_dataloader)):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predictions = output.max(1)
            correct_predictions += torch.sum(predictions == labels).item()
            loss = loss_fn(output, labels)
            losses.append(loss.item())


    return np.mean(losses),correct_predictions / len(test_dataloader.dataset)

def main():
    print("Training Started....")
    N_EPOCHS = 9
    lr = 3e-5
    adam_epsilon = 3e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = LoadDataset(csv_file="train.csv",device=device ,max_length=124)
    test_dataset = LoadDataset(csv_file="test.csv", device=device,max_length=124)

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    model = CustomBertModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss().to(device)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    for epoch in range(N_EPOCHS):
        train_loss = training_step(model, train_dataloader, loss_fn, optimizer)
        loss_eval, accuracy = evaluation(model, test_dataloader, loss_fn)
        scheduler.step()
        print(f"Epoch {epoch+1}/{N_EPOCHS} | Train Loss: {train_loss:.4f} | Eval Loss: {loss_eval:.4f} | Accuracy: {accuracy:.4f}")


    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()

