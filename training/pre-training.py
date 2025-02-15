import torch
from datasets import load_dataset
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Trainer, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset=load_dataset("ACIDE/user-vlm-pt", split="train")

#model_id ="google/paligemma2-3b-ft-docci-448"
model_id ="google/paligemma2-10b-ft-docci-448"

processor = PaliGemmaProcessor.from_pretrained(model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

EOS_TOKEN =processor.tokenizer.eos_token

image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

# for Pretraining
def collate_fn(examples):
  texts = ["<image>" for _ in examples]
  labels= [example['text']+ EOS_TOKEN for example in examples]
  images = [example["image"] for example in examples]
  tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")
  tokens = tokens.to(model.dtype).to(device)
  return tokens


for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = True

for param in model.language_model.parameters():
    param.requires_grad = False

def print_model_parameters(model):
    # Calculate total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate the percentage of trainable parameters
    trainable_percentage = (trainable_params / total_params) * 100

    # Print the results
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {trainable_percentage:.2f}%")

print_model_parameters(model)

args = TrainingArguments(
        num_train_epochs=1,
        remove_unused_columns=False,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,
        warmup_ratio=0.1,
        learning_rate=1e-4,
        weight_decay=1e-5,
        adam_beta2=0.999,
        logging_steps=100,
        optim="adamw_hf",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        push_to_hub=True,
        output_dir="Your output directory",
        overwrite_output_dir=True,
        bf16=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
        #evaluation_strategy="steps",  # Enable evaluation during training
        #eval_steps=100,              # Frequency of evaluation
        #load_best_model_at_end=True   # Load best model based on validation metrics
    )

trainer = Trainer(
        model=model,
        train_dataset=dataset ,
        #eval_dataset=dv,  # Validation dataset
        data_collator=collate_fn,
        args=args
        )

trainer.train()

processor.push_to_hub("YOUR HUGGING FACE direction")
model.push_to_hub("YOUR HUGGING FACE direction")