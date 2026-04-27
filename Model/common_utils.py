import time
import re
import random
import nltk
import emoji
import contractions
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Ensure NLTK data is downloaded if not already
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Using local path for CSV files (parent directory)
BASE_PATH = '../'
RANDOM_STATE = 12345
BATCH_SIZE = 16 
MAX_LEN = 64 
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['negative', 'neutral', 'positive']
num_labels = len(class_names)

# --- Helper Class: TextPreprocessor ---
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        text = text.lower()  # Lowercase
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r"@\w+", "", text)  # Remove mentions
        text = emoji.demojize(text)  # Demojize
        text = contractions.fix(text)  # Fix contractions
        text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
        # Filter out stopwords and lemmatize
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split() if word not in self.stop_words])
        return text

# --- Helper Function: load_project_data ---
def load_project_data():
    train_df = pd.read_csv(BASE_PATH + 'SA_train_cleaned.csv')
    test_df = pd.read_csv(BASE_PATH + 'SA_test_cleaned.csv')

    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_df['sentiment_idx'] = train_df['sentiment'].map(label_map)

    tp = TextPreprocessor()
    train_df['cleaned_text'] = train_df['text'].apply(tp.clean_text)
    test_df['cleaned_text'] = test_df['text'].apply(tp.clean_text)

    train_data_raw, val_data_raw = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df['sentiment_idx'],
        random_state=RANDOM_STATE
    )
    return train_data_raw, val_data_raw, test_df

_STOP_WORDS = set(stopwords.words('english'))

def synonym_replacement(words, n_aug=1):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in _STOP_WORDS]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = []
        for syn in wordnet.synsets(random_word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                synonyms.append(synonym)

        if len(synonyms) > 0:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1

        if num_replaced >= n_aug:
            break
    return new_words

def augment_text_data(df, augment_ratio=0.1, n_aug=1):
    augmented_data = []
    for index, row in df.iterrows():
        text = row['cleaned_text']
        sentiment = row['sentiment_idx']
        augmented_data.append({'cleaned_text': text, 'sentiment_idx': sentiment})
        if random.uniform(0, 1) < augment_ratio:
            for _ in range(n_aug):
                words = text.split()
                augmented_words = synonym_replacement(words)
                augmented_text = ' '.join(augmented_words)
                augmented_data.append({'cleaned_text': augmented_text, 'sentiment_idx': sentiment})
    return pd.DataFrame(augmented_data)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx])
                for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def acc_score(model, dataloader, device=None):
    if device is None:
        device = DEVICE
    model.eval()
    correct_predictions, num_examples = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            _, predicted_labels = torch.max(logits, dim=1)
            num_examples += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()
    final_accuracy = correct_predictions / num_examples
    return final_accuracy

def train_high_accuracy_model(model, train_loader, val_loader, epochs, model_save_path, optimizer_local, scheduler_local, loss_fn_local):
    best_acc = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer_local.zero_grad()

            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=mask)
            loss = loss_fn_local(outputs.logits, labels) # Use loss_fn_local

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevents exploding gradients
            optimizer_local.step()
            scheduler_local.step()

            total_loss += loss.item()

        # Validation Phase
        val_acc = acc_score(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"⭐ New Best Accuracy Reached!")

    print(f"\nTraining Complete. Total Time: {(time.time()-start_time)/60:.2f} min")
    return model

def evaluate_test(model, dataloader, device=None):
    if device is None:
        device = DEVICE
    model.eval()
    correct_preds, num_examples = 0, 0
    test_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            predicted_labels_batch = torch.argmax(logits, dim=1)
            test_predictions.append(predicted_labels_batch.cpu())
            num_examples += labels.size(0)
            correct_preds += (predicted_labels_batch == labels).sum().cpu().item()
    test_accuracy_score = correct_preds / num_examples
    test_predictions_tensor = torch.cat(test_predictions)
    return test_accuracy_score, test_predictions_tensor

def plot_target_structure(labels, fig_title="Sentiment Distribution"):
    """
    Plotting the shares of Dataset labels (Negative, Neutral, Positive).
    """
    labels_count = np.bincount(labels)
    n_obj = labels.shape[0]

    labels_info_share = pd.Series(
        labels_count,
        index=class_names 
    ) / n_obj

    labels_info_share = labels_info_share.sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    colors = ['#66b3ff' if x == 'neutral' else '#ff9999' if x == 'negative' else '#99ff99' for x in labels_info_share.index]

    labels_info_share.plot(kind="bar", color=colors)

    plt.title(fig_title, fontsize=15)
    plt.xlabel("Sentiment Class")
    plt.ylabel("Proportion of Posts")

    plt.xticks(rotation=0)
    plt.tight_layout()

    filename = fig_title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename)
    print(f"Chart saved as {filename}")
