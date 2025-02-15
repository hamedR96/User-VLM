import streamlit as st
import time
from PIL import Image
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

# THIS CHAT APPLICATION HAS NO MEMORY OR HISTORY, MEANING YOUR PREVIOUS MESSAGE IS NOT REMEMBERED BY THE AI.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_id = "ACIDE/User-VLM-10B-Instruct"
#model_id = "ACIDE/User-VLM-3B-Instruct"

if "models_loaded" not in st.session_state:
    torch.mps.empty_cache()
    st.session_state.models_loaded = True
    st.session_state.processor = PaliGemmaProcessor.from_pretrained(model_id)
    st.session_state.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

def response(question, image, model, processor):
    prompt = f"<image> <|im_start|>USER: {question}<|im_end|> ASSISTANT: "
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded


# Enable camera
enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

def response_generator(question, image):

    rsp =response(question, image, model=st.session_state.model, processor=st.session_state.processor)

    for word in rsp.split():
        yield word + " "
        time.sleep(0.05)

st.title("Personalized VQA Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    if picture:
        # Add user message and picture to chat history
        st.session_state.messages.append({"role": "user", "content": f"{prompt} (with picture)"})
        # Display user message and picture in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            #st.image(picture)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = "".join(response_generator(prompt, Image.open(picture)))
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please take a picture before submitting your prompt.")
