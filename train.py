import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
import argparse
import time

def printtime(start_time):
    elapsed_time = time.time() - start_time

    # Convert elapsed time to HH:MM:SS format
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print elapsed time in HH:MM:SS format
    print(f"Elapsed time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a model on image classification.')
    parser.add_argument('data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default=None, help='Set directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Choose architecture (default: vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Set learning rate for model training')
    parser.add_argument('--hidden_units', type=int, default=512, help='Set the hidden units for model training')
    parser.add_argument('--epochs', type=int, default=3, help='Set epochs for model training')
    parser.add_argument('--gpu', action ='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    val_dataset = datasets.ImageFolder(valid_dir, transform = val_test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32)
    
    if hasattr(models, args.arch):
        model_class = getattr(models, args.arch)
        model = model_class(pretrained=True)
    else:
        raise ValueError("Please use the supported architectures in torchvision.models")

    #Freeze the pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, args.hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(args.hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    epochs = args.epochs

    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    model.to(device)
    print("Training on: ", device)
    start_time = time.time()
    print("Start training ...")

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        
        batch = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch+=1
            if batch%10==0:
                print("Training ongoing ...")
                printtime(start_time)

        val_loss = 0
        accuracy = 0
        model.eval()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model.forward(inputs)
                val_loss += criterion(outputs, labels).item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {train_loss:.3f}.. "
            f"Validation loss: {val_loss:.3f}.. "
            f"Validation accuracy: {accuracy:.3f}")
        printtime(start_time)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'arch': args.arch,
        'hidden_units': args.hidden_units,
        'epoch': epochs,
        'loss': train_loss
    }

    checkpoint_path = 'checkpoint.pth'
    if args.save_dir:
        checkpoint_path = args.save_dir + 'checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    main()