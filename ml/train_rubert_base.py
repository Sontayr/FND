import os
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# 1) Модель. Пока tiny, чтобы быстро проверить пайплайн.
# Потом просто поменяешь на "DeepPavlov/rubert-base-cased"
MODEL_NAME = "DeepPavlov/rubert-base-cased"
MAX_LEN = 128
OUT_DIR = "./models/rubert-base-fakenews"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds).tolist()

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix": cm,
    }


def main():
    # 2) Читаем train.tsv
    df = pd.read_csv("train.tsv", sep="\t")

    # Оставляем только нужные колонки и переименуем под HuggingFace
    df = df[["title", "is_fake"]].rename(columns={"title": "text", "is_fake": "label"})

    # (на всякий случай) убираем пустые тексты
    df["text"] = df["text"].astype(str).fillna("")
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # 3) Правильный split: в train и val будут оба класса 0 и 1
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    # 4) Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    # 5) Модель классификации (2 класса: real/fake)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 6) Параметры обучения
    args = TrainingArguments(
        output_dir="./runs",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # 7) Учим
    trainer.train()

    # 8) Итоговая оценка на val
    metrics = trainer.evaluate()
    print("\nFINAL METRICS (val split):", metrics)

    # 9) Сохраняем модель и токенизатор
    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nSaved model to: {OUT_DIR}")


if __name__ == "__main__":
    main()