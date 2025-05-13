import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import os
os.environ["WANDB_DISABLED"] = "true"

# Kiểm tra GPU
print(torch.cuda.is_available())   # Phải là True
print(torch.cuda.get_device_name())  # Phải in ra tên GPU của bạn

# Load dữ liệu
df = pd.read_csv('../path/to/your/data.csv').dropna()
dataset = Dataset.from_pandas(df)

# Chia train/validation (10% val)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")

# Tokenize
max_input_length = 512
max_target_length = 128

def preprocess(example):
    input_text = "Tóm tắt: " + example["văn bản"]
    target_text = example["tóm tắt"]

    # Tokenize input
    model_inputs = tokenizer(
        input_text,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    # Tokenize target (label)
    labels = tokenizer(
        target_text,
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )["input_ids"]

    # Thay pad_token_id trong label bằng -100 để không tính vào loss
    labels = [(label if label != tokenizer.pad_token_id else -100) for label in labels]
    model_inputs["labels"] = labels

    return model_inputs

# Áp dụng tiền xử lý
tokenized_train = train_dataset.map(preprocess, remove_columns=['văn bản', 'tóm tắt'])
tokenized_eval = eval_dataset.map(preprocess, remove_columns=['văn bản', 'tóm tắt'])

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")

from datetime import datetime

training_args = TrainingArguments(
    output_dir="../output/dir",
    run_name=f"vit5-summary-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir='./logs',
    logging_steps=100,
    fp16=True
)

# Huấn luyện với Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# Bắt đầu huấn luyện
trainer.train()

# 👉 Lưu mô hình thủ công sau khi huấn luyện xong
trainer.save_model("./vit5_finetuned")