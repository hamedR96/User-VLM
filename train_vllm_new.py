"""
Fine-Tuning LLMs on Tasks
"""
import argparse
import os
import re

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from data import get_formatted_datasets
from src import (
    TaskType,
    LoraConfig,
    MoleConfig,
    AdaMoleConfig,
    PeftTrainer,
    PeftModelForCausalLM,
)
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration
from transformers import PaliGemmaProcessor

from datasets import load_dataset
from peft import get_peft_model

# from peft import get_peft_model, LoraConfig


transformers.set_seed(0)

class Params():
    def __init__(self):
        pass


def get_default_params():
    args = Params()
    args.model_path = 'meta-llama/Llama-2-7b-hf'
    args.data_path = 'tau/commonsense_qa'
    args.peft_type = 'lora'
    args.lora_rank = 32
    args.target_modules = ['q_proj', 'v_proj']
    args.num_experts = 1
    args.top_k = None
    args.threshold = None
    args.max_length = 256
    args.batch_size = 16
    args.gradient_accumulation_steps = 1
    args.num_train_epochs = 1
    args.learning_rate = 1e-4
    args.lr_scheduler_type = "constant_with_warmup"
    args.warmup_steps = 200
    args.weight_decay = 0.0
    args.aux_loss_coeff = None
    return args

def get_adamole_train_params():
    args = get_default_params()
    args.model_path = 'google/paligemma2-3b-pt-224'
    args.data_path = 'merve/vqav2-small'
    args.peft_type = 'adamole'
    args.lora_rank = 2
    args.target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    args.num_experts = 1
    args.threshold = 0.125
    args.max_length = 256
    args.batch_size = 2
    args.gradient_accumulation_steps = 4
    args.num_train_epochs = 2
    args.learning_rate = 1e-4
    args.lr_scheduler_type = 'constant_with_warmup'
    args.warmup_steps = 200
    args.weight_decay = 0.0
    args.aux_loss_coeff = 1e-3
    return args

def get_mole_train_params():
    args = get_default_params()
    args.model_path = 'google/paligemma2-3b-pt-224'
    args.data_path = 'merve/vqav2-small'
    args.peft_type = 'mole'
    args.lora_rank = 2
    args.target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    args.num_experts = 1
    args.threshold = 0.125
    args.max_length = 256
    args.batch_size = 2
    args.gradient_accumulation_steps = 4
    args.num_train_epochs = 2
    args.learning_rate = 1e-4
    args.lr_scheduler_type = 'constant_with_warmup'
    args.warmup_steps = 200
    args.weight_decay = 0.0
    args.aux_loss_coeff = 1e-3
    return args

def get_lora_train_params():
    args = get_default_params()
    args.model_path = 'google/paligemma2-3b-pt-224'
    args.data_path = 'merve/vqav2-small'
    args.peft_type = 'lora'
    args.lora_rank = 2
    args.target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    args.num_experts = 1
    args.threshold = 0.125
    args.max_length = 256
    args.batch_size = 2
    args.gradient_accumulation_steps = 4
    args.num_train_epochs = 2
    args.learning_rate = 1e-4
    args.lr_scheduler_type = 'constant_with_warmup'
    args.warmup_steps = 200
    args.weight_decay = 0.0
    args.aux_loss_coeff = 1e-3
    return args

# args = get_adamole_train_params()
# args = get_mole_train_params()
args = get_lora_train_params()

print(f'Arguments: {args}')
model_path = args.model_path
data_path = args.data_path
model_name = os.path.basename(model_path).lower()
data_name = os.path.basename(data_path).lower()
peft_type = args.peft_type
num_experts = args.num_experts
max_length = args.max_length
lora_rank = args.lora_rank if peft_type == 'lora' else args.lora_rank // num_experts
lora_alpha = 16
lora_dropout = 0.05
peft_type_name = peft_type
if args.top_k is not None:
    peft_type_name += f'-top{args.top_k}'
if args.threshold is not None:
    threshold_name = int(1 / args.threshold)
    peft_type_name += f'-the{threshold_name}'
output_dir = os.path.join('outputs', re.sub(r'[^0-9a-zA-Z]', '-', f'{model_name}-{peft_type_name}-{data_name}'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and format datasets
# def format_vllm_dataset():

# formatted_datasets = get_formatted_datasets(data_path=data_path, prompt_only=False)

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     model_path,
#     padding_side="left",
#     # add_bos_token=True,
#     add_eos_token=True,
# )
# tokenizer.pad_token = tokenizer.eos_token

# # Tokenize datasets
# tokenize_text = lambda examples: tokenizer(
#     examples["text"],
#     truncation=True,
#     max_length=max_length,
#     # padding=True,
#     # return_tensors="pt",
# )
# tokenized_datasets = formatted_datasets.map(
#     tokenize_text,
#     batched=True,
#     remove_columns=formatted_datasets["train"].column_names,
# )
# print(f'Tokenized datasets: {tokenized_datasets}')

# # Set the data collator
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="pt")

# Load the base model
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     # torch_dtype=torch.bfloat16,
#     # device_map="auto",
# )

def collate_fn(examples):
  texts = ["<image>answer en " + example["question"] for example in examples]
  labels= [example['multiple_choice_answer'] for example in examples]
  images = [example["image"].convert("RGB") for example in examples]
  tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")

  # tokens = tokens.to(DTYPE)
  tokens = tokens.to(device)
  # .to(device)
  return tokens

def collate_fn_one(example):
    texts = ["<image>answer en " + example["question"]]
    labels = [example['multiple_choice_answer']]
    images = [example["image"].convert("RGB")]
    tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")
    
    # tokens = tokens.to(DTYPE)
    # tokens = tokens.to(device)
    # .to(device)
    return tokens


base_model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, device_map="auto")#, quantization_config=bnb_config)
processor = PaliGemmaProcessor.from_pretrained(model_path)
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

# lora_config = LoraConfig(
#     r=8,
#     target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
#     task_type="CAUSAL_LM",
# )

# base_model = get_peft_model(base_model, lora_config)

# for param in base_model.vision_tower.parameters():
#     param.requires_grad = False

# for param in base_model.multi_modal_projector.parameters():
#     param.requires_grad = False

############## temporary dataset

ds = load_dataset('merve/vqav2-small', split="validation")
split_ds = ds.train_test_split(test_size=0.01) # we'll use a very small split for demo
test_ds = split_ds["test"]
# train_ds = split_ds["train"]
train_ds = split_ds["train"].train_test_split(test_size=0.01)["test"]

train_ds = train_ds.map(collate_fn_one)
test_ds = test_ds.map(collate_fn_one)
del ds
del split_ds

# train_ds_ = train_ds.map(collate_fn)

# print(f'Base model  loadedfrom {model_path}')
# print(f'Base model: {base_model}')
print('Model Loaded')

# Get the PEFT model
if peft_type == 'lora':
    peft_config = LoraConfig(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
elif peft_type == 'mole':
    peft_config = MoleConfig(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        num_experts=num_experts,
        top_k=args.top_k,
        threshold=args.threshold,
    )
elif peft_type == 'adamole':
    peft_config = AdaMoleConfig(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        num_experts=num_experts,
        max_threshold=args.threshold,
    )
else:
    raise KeyError(f'Unsupported PEFT type: {peft_type}')

model = PeftModelForCausalLM(base_model, peft_config)
model.enable_input_require_grads()

for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

model.print_trainable_parameters()


# print(f'Model: {model}')

# Set the trainer
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    group_by_length=True,
    remove_unused_columns=False,
    logging_strategy="steps",
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    # save_steps=1000,
    optim="adamw_torch",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=False,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    fp16=True,
    seed=0,
    data_seed=0,
    report_to=["tensorboard"],
    dataloader_pin_memory=False
)
trainer = PeftTrainer(
    # model=model,
    # tokenizer=tokenizer,
    # args=training_args,
    # data_collator=data_collator,
    # train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["validation"],
    # aux_loss_coeff=args.aux_loss_coeff,
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=collate_fn,
    args=training_args,
    aux_loss_coeff=args.aux_loss_coeff,

)
with open(os.path.join(output_dir, 'training_args.json'), 'w') as output_file:
    output_file.write(training_args.to_json_string())

# Train the model
model.config.use_cache = False
trainer.train()
model.config.use_cache = True

# Save the model
trainer.save_model()
trainer.save_state()
