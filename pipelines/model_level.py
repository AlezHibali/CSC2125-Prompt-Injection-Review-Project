from transformers import pipeline
import time
import torch

class ModelLevelPipeline:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.guard = pipeline("text-generation", model="meta-llama/Llama-Guard-3-8B", token=True, device=self.device)
        self.llm = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", token=True, device=self.device)

    def run(self, prompt):
        start = time.time()

        # guard_result = self.guard(prompt)[0]

        # if "unsafe" in guard_result['label'].lower():
        #     return "BLOCKED", time.time() - start

        # output = self.llm(prompt, max_new_tokens=50)[0]['generated_text']

        output = self.guard(prompt, max_new_tokens=50)[0]['generated_text']

        latency = time.time() - start

        return output, latency
    