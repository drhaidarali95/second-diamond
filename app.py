import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 🌌 Comet UI Setup
st.set_page_config(page_title="☄️ Comet - Second Diamond", layout="wide")
st.markdown("<h1 style='text-align:center;'>☄️ Comet: Second Diamond Chat</h1>", unsafe_allow_html=True)

# Load model (tiny for free hosting — swap later for Gemma/LLaMA)
@st.cache_resource
def load_model():
    model_id = "sshleifer/tiny-gpt2"  
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model

tokenizer, model = load_model()

# Conversation memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div style='text-align:right; color:white; background:#007AFF; padding:8px; border-radius:12px; margin:4px;'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left; color:black; background:#E5E5EA; padding:8px; border-radius:12px; margin:4px;'>{msg['content']}</div>", unsafe_allow_html=True)

# User input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div style='text-align:right; color:white; background:#007AFF; padding:8px; border-radius:12px; margin:4px;'>{prompt}</div>", unsafe_allow_html=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f"<div style='text-align:left; color:black; background:#E5E5EA; padding:8px; border-radius:12px; margin:4px;'>{response}</div>", unsafe_allow_html=True)

