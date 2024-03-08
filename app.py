import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load the dataset
df = pd.read_csv('recipe.csv')

# Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Preprocess the data and create training and testing sets
texts = df['title'].tolist()
labels = df['NER'].tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Ensure train and test texts are strings
train_texts = [str(text) for text in train_texts]
test_texts = [str(text) for text in test_texts]

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')

train_labels_encodings = tokenizer(train_labels, truncation=True, padding=True, return_tensors='pt')
test_labels_encodings = tokenizer(test_labels, truncation=True, padding=True, return_tensors='pt')

# Fine-tune the model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.train()

# Train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):  # Adjust the number of epochs as needed
    optimizer.zero_grad()
    outputs = model(**train_encodings, labels=train_labels_encodings)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the model
model.save_pretrained('recipe_model')

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model.generate(**test_encodings)
    print(outputs)
