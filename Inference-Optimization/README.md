# Inrefence Optimization Techniques
This repository contains an example implementation of discrepancy inference optimization methods in PyTorch framework. For comparison several inferencing experiments have provided: 
   - `PyTorch`
   - `TorchScript`
   - `ONNX Runtime`
   - `Dynamic Quantization`

You can see here comparison performance of chosen models for inference on CPU and GPU after several optimizations. Here I use batch size of 1 which is often useful for online inference. Maximum sequence length - [64, 128, 256, 512].

-----------
## CPU results
```
python -m train_single.py
```
Accuracy: **27 %** \
Total elapsed time: **69.03** seconds \
Train 1 epoch: **13.08** seconds


## DDP 2 GPU
```
python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py
```
Accuracy: **19 %** \
Total elapsed time: **97.03** seconds \
Train 1 epoch: **9.79** seconds


## DDP 2 GPU + mixed precision
```
python -m torch.distributed.launch --nproc_per_node=2 train_ddp_mixed.py
```
Accuracy: **15 %** \
Total elapsed time: **70.61** seconds \
Train 1 epoch: **6.52** seconds

---------
These results were close to the idea of N workers would give a speedup of N. A very important note here is that DistributedDataParallel uses an effective batch size of 4*256=1024 so it makes fewer model updates. That’s why it scores a much lower validation accuracy (15% compared to 27% in the baseline).

We'll use batch size of 1 which is useful for online inference. Maximum sequence length - [64, 128, 256, 512]