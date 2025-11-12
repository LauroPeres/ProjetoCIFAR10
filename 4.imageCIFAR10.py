import torch
import torch.nn as nn
import torchvision
import redes  

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainDataset = torchvision.datasets.CIFAR10(root='./data', train=True,
    transform=transform, download=True)

testDataset = torchvision.datasets.CIFAR10(root='./data', train=False,
    transform=transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=100, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=100, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

model = redes.Net()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainLoader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Imprimir a perda média da época
    print(f"Epoch {epoch}, loss: {running_loss/len(trainLoader):.3f}")

# Avaliação no conjunto de teste
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testLoader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Taxa de acerto: {accuracy:.4f} ou {100 * accuracy:.2f}%")

