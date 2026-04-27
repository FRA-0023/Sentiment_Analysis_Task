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
    SentimentDataset, train_high_accuracy_model, evaluate_test,
    RANDOM_STATE, MAX_LEN, BATCH_SIZE, DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, BASE_PATH
)

print("--- Training BERT Base Multilingual Uncased Model ---")
MODEL_NAME_MULTILINGUAL_BERT = "bert-base-multilingual-uncased"

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
tokenizer_multilingual_bert = AutoTokenizer.from_pretrained(MODEL_NAME_MULTILINGUAL_BERT)

training_encodings = tokenizer_multilingual_bert(
    list(training_texts), add_special_tokens=True, max_length=MAX_LEN,
    truncation=True, padding="max_length", return_attention_mask=True
)
validation_encodings = tokenizer_multilingual_bert(
    list(validation_texts), add_special_tokens=True, max_length=MAX_LEN,
    truncation=True, padding="max_length", return_attention_mask=True
)
testing_encodings = tokenizer_multilingual_bert(
    list(testing_texts), add_special_tokens=True, max_length=MAX_LEN,
    truncation=True, padding="max_length", return_attention_mask=True
)

train_dataset = SentimentDataset(training_encodings, training_labels)
val_dataset = SentimentDataset(validation_encodings, validation_labels)
test_dataset = SentimentDataset(testing_encodings, testing_labels)

training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
testing_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

config_multilingual_bert = AutoConfig.from_pretrained(MODEL_NAME_MULTILINGUAL_BERT)
config_multilingual_bert.num_labels = num_labels

model_multilingual_bert = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_MULTILINGUAL_BERT, config=config_multilingual_bert, ignore_mismatched_sizes=True
)
model_multilingual_bert.float()
model_multilingual_bert.to(DEVICE)

optimizer_multilingual_bert = AdamW(model_multilingual_bert.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

total_steps_multilingual_bert = len(training_dataloader) * NUM_EPOCHS
scheduler_multilingual_bert = get_linear_schedule_with_warmup(
    optimizer_multilingual_bert, num_warmup_steps=int(0.1 * total_steps_multilingual_bert), num_training_steps=total_steps_multilingual_bert
)

class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(training_labels), y=training_labels
)
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)

multilingual_bert_model_save_path = BASE_PATH + 'multilingual_bert_sentiment_model_best.pth'

trained_model_multilingual_bert = train_high_accuracy_model(
    model_multilingual_bert, training_dataloader, validation_dataloader, NUM_EPOCHS,
    multilingual_bert_model_save_path, optimizer_multilingual_bert, scheduler_multilingual_bert, loss_fn
)

model_multilingual_bert.load_state_dict(torch.load(multilingual_bert_model_save_path, map_location=DEVICE))
test_acc_multilingual_bert, test_preds_multilingual_bert = evaluate_test(trained_model_multilingual_bert, testing_dataloader)
print(f"\nFINAL TEST ACCURACY (BERT Base Multilingual): {test_acc_multilingual_bert:.4%}")

print(classification_report(testing_labels, test_preds_multilingual_bert, target_names=class_names))

cm_multilingual_bert = confusion_matrix(testing_labels, test_preds_multilingual_bert)
disp_multilingual_bert = ConfusionMatrixDisplay(confusion_matrix=cm_multilingual_bert, display_labels=class_names)
fig_multilingual_bert, ax_multilingual_bert = plt.subplots(figsize=(8, 8))
disp_multilingual_bert.plot(cmap=plt.cm.Blues, ax=ax_multilingual_bert)
plt.title(f'Confusion Matrix - BERT Base Multilingual (Test Accuracy: {test_acc_multilingual_bert:.2%})')
plt.savefig('../multilingual_bert_confusion_matrix.png')
