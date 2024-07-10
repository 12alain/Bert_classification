## Classification des Sentiments avec BERT

## Table des Matières
1. [Introduction](#introduction)
2. [Données](#données)
3. [Entraînement du Modèle](#entraînement-du-modèle)
4. [Évaluation](#évaluation)
5. [Résultats](#résultats)
6. [Implémentation](#implémentation)
    1. [Entraînement du Modèle avec PyTorch](#entrainement-du-modèle-avec-pytorch)
    2. [Interface de Démo avec Gradio](#interface-de-démo-avec-gradio)
    3. [API pour les Tests](#api-pour-les-tests)

## Introduction
Ce projet vise à démontrer l'utilisation de BERT pour la classification des sentiments dans un ensemble de données de textes. BERT est un modèle de langage puissant développé par Google, et nous ajoutons une couche linéaire sur le modèle pré-entraîné pour adapter le modèle à notre tâche spécifique de classification.

## Données
Les données utilisées dans ce projet proviennent de fichiers CSV contenant des textes et des sentiments associés. Les sentiments peuvent être "positif", "négatif" ou "neutre". Les données ont été nettoyées et équilibrées pour améliorer les performances du modèle.

### Nettoyage des Données
Les étapes de nettoyage des données incluent :
- Suppression des caractères spéciaux et des ponctuations.
- Conversion des textes en minuscules.
- Suppression des mots vides (stop words).
- Lemmatisation des mots pour réduire les mots à leur forme de base.

### Rééquilibrage des Classes
Les données sont rééquilibrées pour éviter le déséquilibre des classes en utilisant la technique de suréchantillonnage. Cela implique la duplication des exemples des classes minoritaires pour obtenir une distribution uniforme des classes.

### Description des Données
Les colonnes des fichiers CSV incluent :
- `text` : Le texte à analyser.
- `sentiment` : Le sentiment associé au texte (positif, négatif, neutre).

## Entraînement du Modèle
Nous utilisons le modèle BERT pré-entraîné et ajoutons une couche linéaire pour la classification des sentiments. Le modèle est entraîné sur les données de formation et évalué sur les données de test.

### Préparation des Données
Les textes sont tokenisés en utilisant le tokenizer de BERT et les données sont transformées en tenseurs PyTorch pour l'entraînement.

### Hyperparamètres
- **Nombre d'époques** : 10
- **Taux d'apprentissage** : 3e-5
- **Taux de dropout** : 0.5
- **Taille de batch** : 16 pour l'entraînement, 8 pour l'évaluation

### Entraînement
L'entraînement est effectué en utilisant l'optimiseur AdamW et une fonction de perte de type CrossEntropyLoss. Un scheduler est utilisé pour ajuster le taux d'apprentissage au cours des époques.

## Évaluation
Le modèle est évalué en utilisant les métriques de précision (accuracy) et de perte (loss) sur l'ensemble de test. Les prédictions sont comparées aux étiquettes réelles pour calculer ces métriques.

## Résultats
Les résultats de l'entraînement montrent une diminution de la perte d'entraînement au fil des époques, mais la perte de validation reste élevée, indiquant un possible surapprentissage. Les mesures de précision et de perte sont rapportées pour chaque époque.

### Exemple de Résultats
- **Précision sur l'ensemble de test** : 60%
- **Perte sur l'ensemble de test** : 0.35

Vous pouvez télécharger le dataset utilisé pour ce projet en suivant ce lien : [Télécharger le dataset](https://drive.google.com/file/d/1VwvYyvVpULGqFtLCvvII-fPVq6dRuvMs/view?usp=drive_link)

Les poids du modèle entraîné peuvent être téléchargés ici : [Télécharger les poids du modèle](https://drive.google.com/file/d/1DAaj_jdl4rVNoJauOy3XTDPFMQf03nSH/view?usp=sharing)

## Implémentation

### Entraînement du Modèle avec PyTorch

```python



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



```

# Interface de Démo avec Gradio
```python
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
```

# API pour les Tests
```python
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
import pandas as pd
from tqdm import tqdm
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()
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
class RequetPost(BaseModel):
    text: str
    
@app.get("/")
def read_root():
    return {"hello": "world" }

@app.post("/predict")
def prediction(requet: RequetPost):
    prediction_result = classifier_fn(requet.text)
    print("ok")
    return {
        "predictions": prediction_result
    }

```
