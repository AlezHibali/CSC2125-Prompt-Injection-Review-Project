from transformers import pipeline
import time
import torch

class SystemLevelPipeline:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.detector = pipeline("text-classification", model="vijil/mbert-prompt-injection", token=True, device=self.device)
        self.llm = pipeline("text-generation", model="meta-llama/meta-llama/Meta-Llama-3-8B-Instruct", token=True, device=self.device)

    def run(self, prompt):
        start = time.time()

        result = self.detector(prompt)[0]

        if result['label'] == 'INJECTION':
            return "BLOCKED", time.time() - start

        output = self.llm(prompt, max_new_tokens=50)[0]['generated_text']
        latency = time.time() - start

        return output, latency
