---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: whisper-large-v2-japanese-5k-steps
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# whisper-large-v2-japanese-5k-steps

This model is a fine-tuned version of [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) on the Japanese CommonVoice dataset (v11)..
It achieves the following results on the evaluation set:
- Loss: 0.4200
- Wer: 0.7449

## Model description

This model is finetuned for 5000 steps for research purposes which means that the transcriptions might not be that satisfactory for users.

## Training and evaluation data

- Training Data: CommonVoice (v11) train split
- Validation Data: CommonVoice (v11) Validation split
- Test Data: CommonVoice (v11) Test split

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

### Transcription

```python
from datasets import load_dataset, Audio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
processor = WhisperProcessor.from_pretrained("clu-ling/whisper-large-v2-japanese-5k-steps")
model = WhisperForConditionalGeneration.from_pretrained("clu-ling/whisper-large-v2-japanese-5k-steps").to(device)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")

# load the dataset
commonvoice_eval = load_dataset("mozilla-foundation/common_voice_11_0", "ja", split="validation", streaming=True)
commonvoice_eval = commonvoice_eval.cast_column("audio", Audio(sampling_rate=16000))
sample = next(iter(commonvoice_eval))["audio"]

# features and generate token ids
input_features = processor(sample["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features
predicted_ids = model.generate(input_features.to(device), forced_decoder_ids=forced_decoder_ids)

# decode
transcription = processor.batch_decode(predicted_ids)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)

```

### Framework versions

- Transformers 4.26.0.dev0
- Pytorch 1.13.1
- Datasets 2.8.1.dev0
- Tokenizers 0.13.2
