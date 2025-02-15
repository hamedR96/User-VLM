import gradio as gr
import torch
from pydub import AudioSegment
from transformers import pipeline, PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import numpy as np
from kokoro import KPipeline
import soundfile as sf

# THIS CHAT APPLICATION HAS NO MEMORY OR HISTORY, MEANING YOUR PREVIOUS MESSAGE IS NOT REMEMBERED BY THE AI.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device=device)
model_id = "ACIDE/User-VLM-10B-Instruct"
processor = PaliGemmaProcessor.from_pretrained(model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
kpipeline = KPipeline(lang_code='a',device="mps") # <= make sure lang_code matches voice


def response(question, image, model, processor):
    prompt = f"<image> <|im_start|>USER: {question}<|im_end|> ASSISTANT: "
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded


def transcribe(audio):
    sr, y = audio
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def merge_audio_files(num, output_file):
    combined = AudioSegment.empty()

    for i in range(num):
        audio = AudioSegment.from_wav(f'{i}.wav')  # Load each WAV file
        combined += audio  # Append the audio

    combined.export(output_file, format="mp3")  # Export as MP3
    print(f"Merged audio saved as {output_file}")


def submit(image,audio):
    output_name="output.mp3"
    question=  transcribe(audio)
    res = response(question, image, model, processor)
    generator = kpipeline(
        res, voice='af_heart',  # <= change voice here
        speed=1#, split_pattern=r'\n+'
    )
    #tts = gTTS(text=res, lang ="en")
    num=0
    for i, (gs, ps, audio) in enumerate(generator):
        print(num)
        num+=1
        sf.write(f'{i}.wav', audio, 24000)# save each audio file
    #tts.save(output_name)

    merge_audio_files(num, output_name)

    return output_name

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# User-VLM 360° Demo")
    with gr.Row():
        image_input = gr.Image(sources="webcam")
        audio_input= gr.Audio(sources="microphone")
    audio_output = gr.Audio(label="Generated Audio")
    submit_button = gr.Button("Submit Question")

    # Bind the save function to the button
    #submit_button.click(submit, inputs=[image_input, audio_input], outputs=output_text)
    submit_button.click(submit, inputs=[image_input, audio_input], outputs=audio_output)

# Launch the app
demo.launch()
