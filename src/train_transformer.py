# src/train_transformer.py

import argparse
import os

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from eval_utils import plot_confusion_matrix
from data import get_label_names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer on SST-2."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model name from Hugging Face Hub.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/distilbert-sst2",
        help="Where to save the fine-tuned model and logs.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,          # 只训 1 个 epoch，CPU 会快很多
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=8000,       # 只用 8000 条训练样本
        help="Number of training examples to use (for speed on CPU).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载原始 SST-2 数据
    dataset = load_dataset("glue", "sst2")

    # 2. tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    # 3. 文本 -> token
    def tokenize_function(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    train_dataset = encoded_dataset["train"]
    eval_dataset = encoded_dataset["validation"]

    # 为了在 CPU 上更快训练，只用一部分训练集
    if args.subset_size is not None and len(train_dataset) > args.subset_size:
        train_dataset = train_dataset.shuffle(seed=42).select(
            range(args.subset_size)
        )

    # 4. 评价指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    # 5. 训练参数（用最通用的一些参数，避免版本不兼容）
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",  # 不用 wandb 之类
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # 7. 训练
    train_result = trainer.train()
    print("Training finished.")
    print("Train result:", train_result)

    # 8. 在验证集上做最终评估
    eval_results = trainer.evaluate()
    print("Final evaluation on validation set:", eval_results)

    # 9. 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)

    # 10. 画验证集混淆矩阵
    preds_output = trainer.predict(eval_dataset)
    pred_labels = np.argmax(preds_output.predictions, axis=-1)
    true_labels = preds_output.label_ids

    label_map = get_label_names()
    labels = [label_map[0], label_map[1]]

    os.makedirs("reports/figures", exist_ok=True)
    plot_confusion_matrix(
        y_true=true_labels,
        y_pred=pred_labels,
        labels=labels,
        title="DistilBERT fine-tuned on SST-2 (validation)",
        save_path="reports/figures/confusion_matrix_distilbert_sst2.png",
    )


if __name__ == "__main__":
    main()
