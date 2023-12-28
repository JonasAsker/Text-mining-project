import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer 
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def get_finetune_df(filepath):
    df = pd.read_csv(filepath)
    df['lyrics'] = df['lyrics'].str.lower()
    df = df[['label', 'lyrics']]
    df['text'] = df['lyrics']
    df.drop(columns=['lyrics'])
    replacement_map = {'negative': 0, 'positive': 1, 'neutral': 2}
    df['label'] = df['label'].replace(replacement_map)
    return df

def df_to_dataset(train_df, test_df):
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset_dict = DatasetDict({'train':train_dataset, 'test':test_dataset})
    return dataset_dict

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

accuracy = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", model_max_length=512)

id2label = {0: "negative", 1: "positive", 2: "neutral"}
label2id = {"negative": 0, "positive": 1, "neutral":2}

train_df = get_finetune_df('Data/train.csv')
test_df = get_finetune_df('Data/test.csv')
data = df_to_dataset(train_df, test_df)
tokenized_data = data.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "prajjwal1/bert-tiny", num_labels=3, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir='model',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model('model')