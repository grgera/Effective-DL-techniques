"""
Mostly based on the official pytorch tutorial
Link: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import os 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import time
import torchvision

from utils import setup_for_distributed, save_on_master, is_main_process

GPU_LIST = "0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_LIST

BATCH_SIZE = 256
NUM_WORKERS = 16
EPOCHS = 1
LEARNING_RATE = 0.001
PATH = './cifar_net.pth'


def create_data_loader_cifar10():
    transform = transforms.Compose(
        [
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)                                  
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=trainset, shuffle=True)                                                  
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=testset, shuffle=True)                                         
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, sampler=test_sampler, num_workers=NUM_WORKERS)
    return trainloader, testloader


def train(net, trainloader):
    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    num_of_batches = len(trainloader)
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        trainloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            images, labels = inputs.cuda(), labels.cuda() 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
        print(f'[Epoch {epoch + 1}/{EPOCHS}] loss: {running_loss / num_of_batches:.3f}')
    
    print('Finished Training')


def test(net, PATH, testloader):

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images, labels = images.cuda(), labels.cuda() 
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct // total
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)


if __name__ == '__main__':
    start = time.time()
    
    init_distributed()
    
    trainloader, testloader = create_data_loader_cifar10()
    net = torchvision.models.resnet50(False).cuda()

    # Convert BatchNorm to SyncBatchNorm
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

    start_train = time.time()
    train(net, trainloader)
    end_train = time.time()
    
    if is_main_process:
        save_on_master(net.state_dict(), PATH)
    dist.barrier()

    test(net, PATH, testloader)

    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, \
     Train 1 epoch: {seconds_train:.2f} seconds")