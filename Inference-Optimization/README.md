# Inrefence Optimization Techniques
This repository contains an example implementation of discrepancy inference optimization methods in PyTorch framework. For comparison several inferencing experiments have provided: 
   - `PyTorch`
   - `TorchScript`
   - `ONNX Runtime`
   - `Dynamic Quantization`

You can see here comparison performance of chosen models for inference on CPU and GPU after several optimizations. Here I use batch size of 1 which is often useful for online inference. Maximum sequence length - [64, 128, 256, 512]. As an example, in this set of experiments trained DistilBERT model on Amazon review dataset is used.

-----------
## Google Colab:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grgera/Effective-DL-techniques/Inference-Optimization/blob/main/nlp_inference_optim.ipynb)
-----------
## CPU results
Quantization gave us the most significant improvement in inference speed. After checking validation accuracy, there was the drop from 96.22 to 96.03%. It’s not serious considering model size drop and speedup. If we extend maximum sequence lengths further to 32 and 16, then we can observe that speedup ~ 85% in [16, 32, 64, 128].

| Method | Max len - 512 | Max len - 256 | Max len - 128 | Max len - 64 | Average Speed Up |
| :---:   | :---: | :---: | :---: | :---: | :---: |
| PyTorch | 506ms   | 273ms   |  151ms |  89ms | **0.0x** |
| TorchScript | 507ms  | 263ms   |  145ms | 83ms | **5.2x** |
| ONNX Runtime | 516ms   | 237ms  |  126ms | 72ms | **19.0x** |
| Quantization | 388ms   | 180ms   |  92ms | 50ms | **56.2x** |

---------
## GPU results
GPU support isn’t provided for quantization in Pytorch yet. Although TorchScript wasn't created for speedup improvement, it still yield solid 20% boost versus non-traced model.

FP16 ONNX model showed us very good performance gains. And there are more optimization available, such as disable/enable some fusions and GPU support for quantization.

| Method | Max len - 512 | Max len - 256 | Max len - 128 | Max len - 64 | Average Speed Up |
| :---:   | :---: | :---: | :---: | :---: | :---: |
| PyTorch | 16.1ms   | 12.1ms   |  11.9ms |  11.9ms | **0.0x** |
| TorchScript | 15.9ms  | 11.2ms   |  9.2ms | 8.92ms | **18x** |
| ONNX Runtime | 14.2ms   | 10.0ms  |  8.14ms | 7.57ms | **35x** |

---------
