# Knowledge Distillation
This repository contains an example implementation of knowledge distillation in PyTorch framework. This technique applies to BERT model with Transformers and Hugging Face support.

So, [BERT-base](https://huggingface.co/textattack/bert-base-uncased-SST-2) will be considered as Teacher and [BERT-Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) as Student. In this example Text-Classification as task-specific knowledge distillation task and the Stanford Sentiment Treebank v2 (SST-2) dataset for training will be used. 

 This idea comes from the [DistilBERT](https://arxiv.org/pdf/1910.01108.pdf) paper where it was shown that a student performed better than simply finetuning the distilled language model.

-----------
## Google Colab:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grgera/Effective-DL-techniques/Knowledge-Distillation/blob/main/distill_pipeline.ipynb)
___________
## Results
Distilled BERT-Tiny has **96%** less parameters than the teacher BERT-base and runs **~46.5x** faster while preserving over **90%** of BERT’s performances as measured on the test par of SST2 dataset.

| Model | Parameters | Speed Up | Accuracy |
| :---:   | :---: | :---: | :---: |
| BERT-base | 109M   | 1x   |  93% |
| BERT-tiny | 4M   | 46.5x   |  83% |

