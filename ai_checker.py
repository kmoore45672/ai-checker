import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import math

@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    return model, tokenizer

def calculate_perplexity(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = math.exp(loss)

    return perplexity

st.title("AI Writing Detector")
st.write("Paste your text below to check how likely it is written by AI.")

input_text = st.text_area("Enter text:", height=300)

if st.button("Check Text") and input_text:
    with st.spinner("Analyzing..."):
        model, tokenizer = load_model()
       score = calculate_perplexity(input_text, model, tokenizer)

st.subheader("Results")
if score is None:
    st.warning("Text too short to analyze. Please enter a longer passage.")
else:
    st.write(f"Perplexity Score: **{score:.2f}**")

    if score < 30:
        st.error("⚠️ Likely AI-generated.")
    elif score < 50:
        st.warning("🤔 Possibly AI-generated.")
    else:
        st.success("✅ Likely human-written.")

