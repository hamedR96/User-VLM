import os
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig


#IF YOU ARE USING MULTIPLE GPU UNCOMMENT THIS
#local_rank = int(os.environ.get("LOCAL_RANK", -1))
#if local_rank != -1:
#    device = torch.device("cuda", local_rank)
#else:
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "ACIDE/User-VLM-10B-base"
processor = PaliGemmaProcessor.from_pretrained(model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
)

EOS_TOKEN = processor.tokenizer.eos_token
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

# Configure LoRA
peft_cfg = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_cfg)
model = model.to(device)

# Freeze certain parameters (make sure these attributes exist on your model)
for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

# Load dataset
dataset = load_dataset("ACIDE/user-vlm-instruct", split="train")

def collate_fn(examples):
    texts = [
        f"<image> <|im_start|>USER: {example['question']}<|im_end|> ASSISTANT: "
        for example in examples
    ]
    labels = [example['answer'] + EOS_TOKEN for example in examples]
    images = [example["image"] for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest")
    tokens = tokens.to(model.dtype).to(device)
    return tokens

args = TrainingArguments(
    num_train_epochs=2,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_ratio=0.1,
    learning_rate=2e-5,
    weight_decay=1e-5,
    adam_beta2=0.999,
    logging_steps=1000,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    push_to_hub=True,
    output_dir="User-VLM-10B-Instruct",
    bf16=True,
    dataloader_pin_memory=False,
    #local_rank=local_rank,  # Pass the local rank for DDP
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    data_collator=collate_fn,
    args=args
)

trainer.train()

processor.push_to_hub("YOUR HUGGING FACE direction")
model.push_to_hub("YOUR HUGGING FACE direction")