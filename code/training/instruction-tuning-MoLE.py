from transformers import BitsAndBytesConfig, PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Trainer, \
    TrainingArguments
from peft import get_peft_model, LoraConfig
from MoLE import *
import torch
from datasets import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_LORA = True
USE_QLORA = True
FREEZE_VISION = True

model_id = "ACIDE/User-VLM-10B-base"
# model_id = "ACIDE/User-VLM-3B-base"
processor = PaliGemmaProcessor.from_pretrained(model_id)

if USE_LORA or USE_QLORA:
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    else:
        bnb_config = None

    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto",
                                                              quantization_config=bnb_config,
                                                              torch_dtype=torch.bfloat16)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto").to(device)


class Args:
    dense_moe = False
    lora_rank = 16
    lora_alpha = 16
    num_experts = 3


args = Args()

for i in range(len(model.language_model.model.layers)):
    original_mlp = model.language_model.model.layers[i].mlp
    model.language_model.model.layers[i].mlp = LoRA_MOE_LM(args=args,
                                                           lora_rank=args.lora_rank,
                                                           lora_alpha=args.lora_alpha,
                                                           num_experts=args.num_experts,
                                                           original_module=original_mlp).to(torch.bfloat16)
model = model.to(device)

if FREEZE_VISION:
    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

model.print_trainable_parameters()

EOS_TOKEN =processor.tokenizer.eos_token  # Ensure the EOS token is defined
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        weight_decay=1e-5,
        adam_beta2=0.999,
        logging_steps=500,
        optim="adamw_hf",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        push_to_hub=True,
        output_dir="your output directory",
        bf16=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
        #evaluation_strategy="steps",  # Enable evaluation during training
        #eval_steps=500,              # Frequency of evaluation
        #load_best_model_at_end=True   # Load best model based on validation metrics
    )

trainer = Trainer(
        model=model,
        train_dataset=dataset ,
        #eval_dataset=test_dataset,  # Validation dataset
        data_collator=collate_fn,
        args=args
        )

trainer.train()
processor.push_to_hub("your hugging face direction")