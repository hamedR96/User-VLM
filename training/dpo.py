from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import torch
from datasets import load_dataset
import os
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer

#IF YOU ARE USING MULTIPLE GPU UNCOMMENT THIS
#local_rank = int(os.environ.get("LOCAL_RANK", -1))
#if local_rank != -1:
#    device = torch.device("cuda", local_rank)
#else:
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "ACIDE/User-VLM-10B-Instruct"
processor = PaliGemmaProcessor.from_pretrained(model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
)


train_dataset = load_dataset("ACIDE/user-vlm-lazy-dpo", split="train")


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'v_proj', 'q_proj'],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model=model.to(device)

training_args = DPOConfig(
    num_train_epochs=1,
    learning_rate=2e-6,
    per_device_train_batch_size=1,
    #do_eval=True,
    #per_device_eval_batch_size=1,
    adam_epsilon=1e-08,
    #lr_scheduler_type="linear",
    warmup_ratio=0.1,
    seed=42,
    logging_steps=1000,
    save_steps=500,
    save_strategy="steps",
    output_dir="your output directory",
    #gradient_checkpointing=True,
    bf16=True,
    remove_unused_columns=False,
)

dpo_trainer = DPOTrainer(
    model,
    #ref_model,
    args=training_args,
    #beta=training_args.beta,
    train_dataset=train_dataset,
    #eval_dataset=dataset["test"],
    processing_class=processor,
    #max_length=training_args.max_length,
    #max_prompt_length=training_args.max_prompt_length,
    peft_config=peft_config,
)

dpo_trainer.train()

processor.push_to_hub("Your HUGGING FACE direction")
dpo_trainer.push_to_hub("Your HUGGING FACE direction")