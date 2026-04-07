from transformers import pipeline, BitsAndBytesConfig
import time
import torch

class BaselinePipeline:
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        print(f"Using device: {'GPU' if use_cuda else 'CPU'}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_quant_type="nf4",           
            bnb_4bit_use_double_quant=True       
        )

        self.pipe = pipeline(
            "text-generation", 
            # model="meta-llama/Meta-Llama-3-8B-Instruct", 
            model="microsoft/deberta-v3-base",
            token=True, 
            device_map="auto", 
            model_kwargs={"quantization_config": quantization_config}
        )

    def run(self, prompt):

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.pipe.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        start = time.time()
        
      
        with torch.no_grad():
            output_list = self.pipe(
                formatted_prompt, 
                max_new_tokens=50,   
                do_sample=False,      
                truncation=True,     
                pad_token_id=self.pipe.tokenizer.eos_token_id,
                return_full_text=False 
            )
            
        output = output_list[0]['generated_text'].strip()
        latency = time.time() - start
        
        return output, latency
    