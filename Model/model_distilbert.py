import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig

from common_utils import (
    load_project_data, augment_text_data, class_names, num_labels,
    SentimentDataset, train_high_accuracy_model, evaluate_test
)

# --- CONFIGURATION & HYPERPARAMETERS ---
BASE_PATH = '../'
RANDOM_STATE = 12345
BATCH_SIZE = 16 
MAX_LEN = 64 
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("--- Training DistilBERT Base Uncased Model ---")
MODEL_NAME_DISTILBERT = "distilbert-base-uncased"

# Load and preprocess original data
train_data_original, val_data_original, test_df_unused = load_project_data()

# Apply augmentation to the training data
augmented_train_df = augment_text_data(train_data_original, augment_ratio=0.5, n_aug=1)
train_data = augmented_train_df 

# Re-extract texts and labels from the augmented training data
texts_data = train_data["cleaned_text"].values.astype("U")
labels_data_encoded = train_data["sentiment_idx"].values

# Data splitting
(training_texts, validation_testing_texts,
 training_labels, validation_testing_labels) = train_test_split(
    texts_data, labels_data_encoded, train_size=0.8,
    random_state=RANDOM_STATE, stratify=labels_data_encoded
)
(validation_texts, testing_texts,
 validation_labels, testing_labels) = train_test_split(
    validation_testing_texts, validation_testing_labels, train_size=0.5,
    random_state=RANDOM_STATE, stratify=validation_testing_labels
)

# Tokenization
tokenizer_distilbert = AutoTokenizer.from_pretrained(MODEL_NAME_DISTILBERT)

training_encodings = tokenizer_distilbert(
    list(training_texts), add_special_tokens=True, max_length=MAX_LEN,
    truncation=True, padding="max_length", return_attention_mask=True
)
validation_encodings = tokenizer_distilbert(
    list(validation_texts), add_special_tokens=True, max_length=MAX_LEN,
    truncation=True, padding="max_length", return_attention_mask=True
)
testing_encodings = tokenizer_distilbert(
    list(testing_texts), add_special_tokens=True, max_length=MAX_LEN,
    truncation=True, padding="max_length", return_attention_mask=True
)

train_dataset = SentimentDataset(training_encodings, training_labels)
val_dataset = SentimentDataset(validation_encodings, validation_labels)
test_dataset = SentimentDataset(testing_encodings, testing_labels)

training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
testing_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

config_distilbert = AutoConfig.from_pretrained(MODEL_NAME_DISTILBERT)
config_distilbert.num_labels = num_labels

model_distilbert = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_DISTILBERT, config=config_distilbert, ignore_mismatched_sizes=True
)
model_distilbert.float()
model_distilbert.to(DEVICE)

optimizer_distilbert = AdamW(model_distilbert.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

total_steps_distilbert = len(training_dataloader) * NUM_EPOCHS
scheduler_distilbert = get_linear_schedule_with_warmup(
    optimizer_distilbert, num_warmup_steps=int(0.1 * total_steps_distilbert), num_training_steps=total_steps_distilbert
)

class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(training_labels), y=training_labels
)
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

distilbert_model_save_path = BASE_PATH + 'distilbert_sentiment_model_best.pth'

trained_model_distilbert = train_high_accuracy_model(
    model_distilbert, training_dataloader, validation_dataloader, NUM_EPOCHS, 
    distilbert_model_save_path, optimizer_distilbert, scheduler_distilbert, loss_fn
)

model_distilbert.load_state_dict(torch.load(distilbert_model_save_path, map_location=DEVICE))
test_acc_distilbert, test_preds_distilbert = evaluate_test(trained_model_distilbert, testing_dataloader)
print(f"\nFINAL TEST ACCURACY (DistilBERT): {test_acc_distilbert:.4%}")

print(classification_report(testing_labels, test_preds_distilbert, target_names=class_names))

cm_distilbert = confusion_matrix(testing_labels, test_preds_distilbert)
disp_distilbert = ConfusionMatrixDisplay(confusion_matrix=cm_distilbert, display_labels=class_names)
fig_distilbert, ax_distilbert = plt.subplots(figsize=(8, 8))
disp_distilbert.plot(cmap=plt.cm.Blues, ax=ax_distilbert)
plt.title(f'Confusion Matrix - DistilBERT (Test Accuracy: {test_acc_distilbert:.2%})')
plt.savefig('../distilbert_confusion_matrix.png')
