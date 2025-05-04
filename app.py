import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Title
st.set_page_config(page_title="Meditron Chat", layout="centered")
st.title("🩺 Meditron-7B Medical Chatbot")
st.markdown("Test the base behavior of [malhajar/meditron-7b-chat](https://huggingface.co/malhajar/meditron-7b-chat)")

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 64, 1024, 512, 64)

# Load tokenizer and model
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("malhajar/meditron-7b-chat")
    model = AutoModelForCausalLM.from_pretrained(
        "malhajar/meditron-7b-chat",
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

# Load model
pipe = load_model()

# User input
user_input = st.text_area("Enter your medical question:", height=100)

if st.button("Ask") and user_input:
    with st.spinner("Generating answer..."):
        prompt = f"You are a medical assistant. Answer this question accurately and concisely.\n\nQuestion: {user_input}\nAnswer:"
        output = pipe(prompt, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
        answer = output[0]['generated_text'].split("Answer:")[-1].strip()
        st.markdown("### 🧠 Answer:")
        st.write(answer)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [Transformers](https://huggingface.co/transformers/) and Streamlit")
