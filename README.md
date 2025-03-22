---
base_model: TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
library_name: peft
---

![LOGO](images/EconoMind-logo.jpg)

# EconoMind

**EconoMind** is a fine-tuned version of `Mistral-7B-Instruct-v0.2-GPTQ` optimized for finance and economics. It is based on the `Mistral-7B` architecture, which is a dense model with 7 billion parameters. The fine-tuning process was implemented by `Quantized Low-Rank Adaptation (QLoRA)`, which quantize whole model's paramters and adds adapters layers inside transformer in the base model rather than retraining the entire model from scratch or using full fune-tuning. `Supervised Fune-Tuning (SFT)` was used to tune the model, which represent pair of data: first as user input, second as LLM respond that we what our model to respond. **EconoMind** was trained on `Josephgflowers/Finance-Instruct-500k` dataset from huggingface, however only **`5000`** of samples were used for **training_dataset**. Furthermore dataset doesn`t have eval/test dataset so model was trained and evaluated on the same **training_dataset**.

## Model Details

### Model Description

- **Developed by:** Mark Mazur
- **Shared by:** Mark228zzz
- **Model type:** Causal Language Model (LLM)
- **Language(s) (NLP):** English
- **License:** [MIT License]
- **Fine-tuned from model:** [TheBloke/Mistral-7B-Instruct-v0.2-GPTQ](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k)

### Model Sources

- **GitHub Repository:** [GitHub](https://github.com/Mark228zzz/EconoMind)
- **HuggingFace Repository:** [HuggingFace](https://huggingface.co/Markoskokos/EconoMind)
- **medium Article:** [Article](https://medium.com/@mark.mazur007) !

## Uses

### Direct Use

**EconoMind** can generate text based answer on your finance or ecomonics questions. It can assist with aconomic analysis, answering policy-related questions, and summarazing financial topics. Useful for educational purposes, researches, and general knowledge retrieval.

## Bias, Risks, and Limitations

- **Bias:** The model may reflect biases in the training data, particularly in economic policy discussions.
- **Hallucinations:** It might generate incorrect or misleading information.
- **Outdated Data:** It fine-tuned on past datasets, it may not reflect current economic conditions.
- **Not Legally or Financially Reliable:** It should not be used for making legally binding decisions or investment choices.

### Recommendations

- Users should fact-check critical outputs before relying on them for real-world decisions.
- It’s best used as a supplementary tool rather than a primary decision-making system.
- Regular updates and re-training on recent data can improve its accuracy over time.

## How to Get Started with the Model

First of all you will need to login in [huggingface](https://huggingface.co/) and [WandB](https://wandb.ai/) and get your `huggingface_token` and `wandb_api_key`

### 1. Install Environment and Set Token and API Key
```bash
python -m venv YOUR_ENV_NAME
sourse YOUR_ENV_NAME/bin/activate
export HUGGUNGFACE_TOKEN="YOUR TOKEN"
export WANDB_API_KEY="YOUR API KEY"
```

### 2. Install Required Dependecies

```bash
pip install -r requirements.txt
```

### 3. Run usage.py
```bash
python usage.py
```

### Run training the model [OPTIONAL]
Modify `train.py` hyperparams as you want and run it to train.
**Require a lot of GPU and CPU memory! Make sure that you have it!**
```bash
python train.py
```

### Results

Example of answer:

**User:** `Explain the difference between fiscal and monetary policy.`

**EconoMind:** `Fiscal policy and monetary policy are two primary tools that governments and central banks use to manage the economy. While they both aim to influence economic conditions, they operate through different mechanisms. Fiscal policy refers to the use of the government's budget and taxation policies to influence the economy.`

## Hardware overview

- **Hardware Type:** One NVIDIA A100-SXM4-40GB
- **Hours used:** 3
- **Cloud Provider:** Google Deep Learning VM
- **Compute Region:** North America, North East 1

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Carbon Emitted:** 0.02 kg CO2 eq
