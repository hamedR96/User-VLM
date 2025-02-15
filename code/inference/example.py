import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "ACIDE/User-VLM-10B-Instruct"

processor = PaliGemmaProcessor.from_pretrained(model_id)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)


def generate_answer(question, image, model, processor):
    prompt = f"<image> <|im_start|>USER: {question}<|im_end|> ASSISTANT: "
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded


question="who is Elon Musk?"


url="https://media.istockphoto.com/id/1282695693/photo/little-boy-sitting-on-chair-at-the-table.jpg?s=612x612&w=0&k=20&c=FP4Cg5qI1gomYcQgaiCkIWWPajhDu7c9Ev2eb9kURhY="

image = load_image(url)

answer=generate_answer(question, image, model=model, processor=processor)

print(answer)