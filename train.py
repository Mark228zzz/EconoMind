from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import huggingface_hub
import wandb
import torch
import os

# Define device (the project require CUDA)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def logins():
    '''
    Login into hugging face and WandB. Connect to your env
    and load environment variables:
    export HUGGUNGFACE_TOKEN="YOUR TOKEN"
    export WANDB_API_KEY="YOUR API KEY"
    Otherwise uncomment function #logins() in main() function to
    set tokens manually.
    '''
    huggingface_hub.login()
    wandb.login()

def load_models():
    print('------     LOAD MODELS     ------')
    # Load the LLM
    qmodel_name = 'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ'
    qmodel = AutoModelForCausalLM.from_pretrained(
        qmodel_name,
        device_map='auto',
        trust_remote_code=False,
        revision='main'
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(qmodel_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return qmodel, tokenizer, data_collator

def use_model(model, tokenizer, prompt):
    # Set pad_token
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    prompt = f'[INST] {prompt} [/INST]'

    print(f'\nYou: {prompt}\n')

    # Tokenize prompt and attach to the device
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get generated response in tokens
    output = model.generate(input_ids=inputs['input_ids'], max_new_tokens=500)

    # Detokenize and print response
    print(f'LLM: {tokenizer.batch_decode(output)[0]}\n')

def setting_lora(qmodel):
    print('------     SETTING LoRA     ------')
    config = LoraConfig(
        r=wandb.config.rank,
        lora_alpha=wandb.config.lora_alpha,
        lora_dropout=wandb.config.lora_dropout,
        bias='none',
        target_modules=['q_proj'],
        task_type='CAUSAL_LM'
    )

    qlora_model = get_peft_model(qmodel, config)

    qlora_model.print_trainable_parameters()

    return qlora_model

def preprocess_dataset(tokenizer):
    print('------     PREPROCESS DATASET     ------')
    '''
    Preprocess dataset for Supervised Fine-Tuning (SFT) and
    tokenizer whole dataset
    '''
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("Josephgflowers/Finance-Instruct-500k")

    dataset = dataset['train'].select(range(wandb.config.sample_numbers))

    # Format the dataset for Supervised Fine-Tuning (SFT)
    def format_dataset(examples):
        formatted = f'[INST] {examples["user"]} [/INST] {examples["assistant"]}'

        return {'text': formatted}

    formatted_dataset = dataset.map(format_dataset)

    # Tokenize dataset
    def tokenize(examples):
        encodings = tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )
        encodings['labels'] = encodings['input_ids']

        return encodings

    # Tokenize formatted dataset
    tokenized_dataset = formatted_dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['user', 'assistant', 'system', 'labels'])

    return tokenized_dataset

def set_hyperparams():
    # Get full path of results folder to save the models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'results')

    return TrainingArguments(
        output_dir=output_path,
        per_device_eval_batch_size=wandb.config.batch_size,
        per_device_train_batch_size=wandb.config.batch_size,
        num_train_epochs=wandb.config.epochs,
        save_strategy='epoch',
        eval_strategy='epoch',
        logging_strategy='epoch',
        logging_steps=1,
        optim='paged_adamw_8bit',
        weight_decay=0.01,
        warmup_steps=2,
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        fp16=True,
        report_to='wandb',
        run_name='Supervised Fine Tuning using QLoRA'
    )

def fine_tune(qlora_model, training_args, tokenized_dataset, data_collator):
    print('------     FINE-TUNING     ------')
    # Define a trainer
    trainer = Trainer(
        model=qlora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    qlora_model.config.use_cache = False

    # Fine-Tune the model
    trainer.train()

    qlora_model.config.use_cache = True

def save(qlora_model, name):
    qlora_model.save_pretrained(f'./FT_models/{name}')

def main():
    #logins()

    # Init WandB application
    wandb.init(
        project='fine-tuning',
        entity='for-riverside-school-s',
        name='Supervised Fine Tuning using QLoRA'
    )

    # Set hyperparams
    # YOU CAN MODIFY HYPERPARAMETERS HERE
    wandb.config.update({
        'lr': 2e-4,
        'batch_size': 4,
        'epochs': 30,
        'rank': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'sample_numbers': 2000,
    })

    # Load quantized model, tokenizer and data_collator
    qmodel, tokenizer, data_collator = load_models()

    # Set the model to train mode and enable gradient checkpointing
    qmodel.train()
    qmodel.gradient_checkpointing_enable()
    qmodel = prepare_model_for_kbit_training(qmodel)

    # Get prepated QLoRA model
    qlora_model = setting_lora(qmodel)

    # Get full tokenized dataset
    tokenized_dataset = preprocess_dataset(tokenizer)

    training_args = set_hyperparams()

    # Start to fine-tuning the model
    fine_tune(qlora_model, training_args, tokenized_dataset, data_collator)

    # Save the model
    save(qlora_model, 'EconoMind_v2')

    # Finish WandB
    wandb.finish()

if __name__ == '__main__':
    main()
