---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: whisper-large-v2-japanese-24h
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# whisper-large-v2-japanese-24h

This model is a fine-tuned version of [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4200
- Wer: 0.7449

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 50
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 5000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 0.0111        | 7.63  | 1000 | 0.3210          | 0.7888 |
| 0.0007        | 15.27 | 2000 | 0.3585          | 0.7478 |
| 0.0003        | 22.9  | 3000 | 0.3937          | 0.7432 |
| 0.0002        | 30.53 | 4000 | 0.4123          | 0.7443 |
| 0.0002        | 38.17 | 5000 | 0.4200          | 0.7449 |


### Framework versions

- Transformers 4.26.0.dev0
- Pytorch 1.13.1
- Datasets 2.8.1.dev0
- Tokenizers 0.13.2
