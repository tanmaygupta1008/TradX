"""Quick HF connectivity test — run: python test_hf.py"""
import os, traceback
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN    = os.getenv("HF_TOKEN", "")
HF_MODEL    = os.getenv("HF_MODEL",    "meta-llama/Llama-3.1-8B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "together")

print(f"Token    : {'SET (' + HF_TOKEN[:8] + '...)' if HF_TOKEN else 'NOT SET'}")
print(f"Model    : {HF_MODEL}")
print(f"Provider : {HF_PROVIDER}")
print("-" * 50)

# Llama 3.1 chat template
formatted_prompt = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant. Reply with valid JSON only.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    'Respond with exactly: {"status": "ok"}<|eot_id|>'
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

try:
    client = InferenceClient(provider=HF_PROVIDER, token=HF_TOKEN)
    result = client.text_generation(
        formatted_prompt,
        model=HF_MODEL,
        max_new_tokens=30,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
    )
    print("SUCCESS:", repr(result))
except Exception as e:
    print(f"ERROR TYPE : {type(e).__name__}")
    print(f"ERROR MSG  : {e}")
    print("\nFULL TRACE:")
    traceback.print_exc()
