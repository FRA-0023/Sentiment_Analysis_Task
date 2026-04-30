import torch
from common_utils import (
    load_project_data, augment_text_data, class_names
)

# --- CONFIGURATION & HYPERPARAMETERS ---
BASE_PATH = '../'
RANDOM_STATE = 12345
BATCH_SIZE = 16 
MAX_LEN = 64 
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FEATURES = 5000
ALPHA_PARAM = 1.0
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("Training Multinomial Naive Bayes Model")

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
    texts_data,
    labels_data_encoded,
    train_size=0.8,
    random_state=RANDOM_STATE,
    stratify=labels_data_encoded
)
(validation_texts, testing_texts,
 validation_labels, testing_labels) = train_test_split(
    validation_testing_texts,
    validation_testing_labels,
    train_size=0.5,
    random_state=RANDOM_STATE,
    stratify=validation_testing_labels
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
X_train_tfidf = vectorizer.fit_transform(training_texts)
X_test_tfidf = vectorizer.transform(testing_texts)

# Initialize and Train Multinomial Naive Bayes Model
mnb_model = MultinomialNB(alpha=ALPHA_PARAM)
mnb_model.fit(X_train_tfidf, training_labels)

# Evaluate Multinomial Naive Bayes Model
mnb_predictions = mnb_model.predict(X_test_tfidf)

# Calculate accuracy
mnb_test_accuracy = (mnb_predictions == testing_labels).mean()
print(f"\nFINAL TEST ACCURACY (Multinomial Naive Bayes): {mnb_test_accuracy:.4%}")

print(classification_report(testing_labels, mnb_predictions, target_names=class_names))

cm_mnb = confusion_matrix(testing_labels, mnb_predictions)
disp_mnb = ConfusionMatrixDisplay(confusion_matrix=cm_mnb, display_labels=class_names)
fig_mnb, ax_mnb = plt.subplots(figsize=(8, 8))
disp_mnb.plot(cmap=plt.cm.Blues, ax=ax_mnb)
plt.title(f'Confusion Matrix - Multinomial Naive Bayes (Test Accuracy: {mnb_test_accuracy:.2%})')
plt.savefig('../naive_bayes_confusion_matrix.png')
