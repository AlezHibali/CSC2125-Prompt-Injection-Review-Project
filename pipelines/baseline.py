from transformers import pipeline
import time
import torch

class BaselinePipeline:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")

        self.pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", token=True, device=self.device)

    def run(self, prompt):
        start = time.time()
        output = self.pipe(prompt, max_new_tokens=50)[0]['generated_text']
        latency = time.time() - start
        return output, latency
