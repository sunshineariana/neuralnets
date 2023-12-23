---
license: mit
base_model: gpt2
tags:
- generated_from_keras_callback
model-index:
- name: sunshineariana/gpt2-finetuned-wikitext2
  results: []
---

<!-- This model card has been generated automatically according to the information Keras had access to. You should
probably proofread and complete it, then remove this comment. -->

# sunshineariana/gpt2-finetuned-wikitext2

This model is a fine-tuned version of [gpt2](https://huggingface.co/gpt2) on an unknown dataset.
It achieves the following results on the evaluation set:
- Train Loss: 6.0668
- Validation Loss: 5.9870
- Epoch: 1

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- optimizer: {'name': 'AdamWeightDecay', 'learning_rate': 2e-05, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False, 'weight_decay_rate': 0.01}
- training_precision: float32

### Training results

| Train Loss | Validation Loss | Epoch |
|:----------:|:---------------:|:-----:|
| 6.7390     | 6.3213          | 0     |
| 6.0668     | 5.9870          | 1     |


### Framework versions

- Transformers 4.36.2
- TensorFlow 2.15.0
- Datasets 2.16.0
- Tokenizers 0.15.0
