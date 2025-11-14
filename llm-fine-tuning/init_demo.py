import pandas as pd
import re
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create a noisy dataset
data_dict = {
    "text": [
        "  The staff was very kind and attentive to my needs!!!  ",
        "The waiting time was too long, and the staff was rude. Visit us at http://hospitalreviews.com",
        "The doctor answered all my questions...but the facility was outdated.   ",
        "The nurse was compassionate & made me feel comfortable!! :) ",
        "I had to wait over an hour before being seen.  Unacceptable service! #frustrated",
        "The check-in process was smooth, but the doctor seemed rushed. Visit https://feedback.com",
        "Everyone I interacted with was professional and helpful.  "
    ],
    "label": ["positive", "negative", "neutral", "positive", "negative", "neutral", "positive"]
}

# Convert dataset to a DataFrame
data = pd.DataFrame(data_dict)

# Clean the text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

data["cleaned_text"] = data["text"].apply(clean_text)

# Convert labels to integers
label_map = {"positive": 0, "neutral": 1, "negative": 2}
data["label"] = data["label"].map(label_map)

# Tokenize the cleaned text
data['tokenized'] = data['cleaned_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Pad or truncate to fixed length (e.g., 128 tokens)
data['padded_tokenized'] = data['tokenized'].apply(
    lambda x: x + [tokenizer.pad_token_id] * (128 - len(x)) if len(x) < 128 else x[:128]
)

# Preview cleaned and labeled data
print(data[['cleaned_text', 'label', 'padded_tokenized']].head())

from sklearn.model_selection import train_test_split

# Split data: 70% training, 15% validation, 15% test
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Training Size: {len(train_data)}, Validation Size: {len(val_data)}, Test Size: {len(test_data)}")

from datasets import Dataset


# Convert DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True, max_length=128)

# Tokenize the dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["text", "cleaned_text"])
val_dataset = val_dataset.remove_columns(["text", "cleaned_text"])
test_dataset = test_dataset.remove_columns(["text", "cleaned_text"])

# Convert labels to int if they are not already
train_dataset = train_dataset.map(lambda x: {"label": int(x["label"])})
val_dataset = val_dataset.map(lambda x: {"label": int(x["label"])})
test_dataset = test_dataset.map(lambda x: {"label": int(x["label"])})

# Print a sample to confirm input_ids exist
print(train_dataset[0])

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    output_dir='./results',
    evaluation_strategy="epoch",
    logging_strategy="epoch",  
    logging_dir='./logs',  
    save_strategy="epoch",  
    load_best_model_at_end=True 
)

# **Explain 'evaluation_strategy':** 
# This determines when the model is evaluated. 'Epoch' evaluates the model after each training epoch.

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.with_format("torch", columns=["input_ids", "attention_mask", "label"]),
    eval_dataset=val_dataset.with_format("torch", columns=["input_ids", "attention_mask", "label"])
)

# Start training
trainer.train()

from sklearn.metrics import accuracy_score, f1_score

# Generate predictions
test_dataset = test_dataset.with_format("torch", columns=["input_ids", "attention_mask", "label"])
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = test_dataset["label"]

# Calculate metrics
accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average="weighted")

print(f"Accuracy: {accuracy}, F1 Score: {f1}")

# **Explain metric importance**:
# High F1 scores indicate balanced performance across all classes, crucial in tasks like sentiment analysis.

# Save the model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")
print("Model saved successfully!")