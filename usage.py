from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftConfig
import torch

# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_models(ft_model_path):
    # Load the quantized LLM
    ft_model_name = 'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ'
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_model_name,
    )

    # Load Fine-Tuned model
    config = PeftConfig.from_pretrained(ft_model_path)
    ft_model = get_peft_model(ft_model, config)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ft_model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    return ft_model, tokenizer

def usage(ft_model_path, prompt, max_new_tokens=500):
    ft_model, tokenizer = load_models(ft_model_path)

    ft_model.to(device)

    tokenizer.pad_token = tokenizer.eos_token

    prompt = f'[INST] {prompt} [/INST]'

    # Transform prompt into tokens
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get outputs in tokens
    output = ft_model.generate(input_ids=inputs['input_ids'], max_new_tokens=max_new_tokens)

    # Print readable text
    print(f'\nEconoMind: {tokenizer.batch_decode(output)[0]}\n')

def main():
    usage(
        ft_model_path='./FT_models/EconoMind',
        prompt='''Explain the difference between fiscal and monetary policy''',
        max_new_tokens=200
    )

if __name__ == '__main__':
    main()
