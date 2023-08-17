"""
Mostly based on the official pytorch tutorial
Link: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torchvision

BATCH_SIZE = 256
NUM_WORKERS = 10
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=NUM_WORKERS)
    return trainloader, testloader


def train(net, trainloader):
    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    num_of_batches = len(trainloader)
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

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
    net.load_state_dict(torch.load(PATH))

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


if __name__ == '__main__':
    start = time.time()
    
    trainloader, testloader = create_data_loader_cifar10()
    net = torchvision.models.resnet50(False).cuda()

    start_train = time.time()
    train(net, trainloader)
    end_train = time.time()

    torch.save(net.state_dict(), PATH)

    test(net, PATH, testloader)

    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, \
     Train 1 epoch: {seconds_train:.2f} seconds")