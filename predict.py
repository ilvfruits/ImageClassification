import torch
from torchvision import datasets, transforms, models
from torch import nn, optim, FloatTensor
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict image classes.')
    parser.add_argument('path_to_image', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path for the model checkpoint file (.pth)')
    parser.add_argument('--topk', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--mapfile', type=str, default=None, help='Path to category names mapping file')
    parser.add_argument('--gpu', action ='store_true', help='Use GPU for inference')
 
    args = parser.parse_args()

    display(args.path_to_image, args.checkpoint, args.topk, args.mapfile, args.gpu)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    tensor_image = val_test_transforms(pil_image)
    numpy_image = np.array(tensor_image)
    numpy_image = numpy_image.transpose((1, 2, 0))
    
    return numpy_image

def imshow(image, ax=None, title=None):
   #Imshow for Tensor.
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((2, 0, 1))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model_checkpoint, topk=5, mapfile=None, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    checkpoint = torch.load(model_checkpoint)

    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']

    if hasattr(models, arch):
        model_class = getattr(models, arch)
        model = model_class()
    else:
        raise ValueError("Please use the supported architectures in torchvision.models")

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
    
    image = process_image(image_path)
    image = image.transpose((2, 0, 1))
    image = torch.FloatTensor(image).unsqueeze(0)

    if gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
    
    probabilities = torch.exp(outputs)
    top_probabilities, top_indices = probabilities.topk(topk, dim=1)

    if mapfile:
        with open(mapfile, 'r') as f:
            cat_to_name = json.load(f)
            top_classes = [cat_to_name[str(idx+1)] for idx in top_indices.tolist()[0]]
    else:
        top_classes = [str(idx+1) for idx in top_indices.tolist()[0]]
    
    return (top_probabilities.tolist()[0], top_classes)
    
def display(image_path, model_checkpoint, topk=5, mapfile=None, gpu=False):
    top_probs, top_classes = predict(image_path, model_checkpoint, topk, mapfile, gpu)
    print(top_probs)
    print(top_classes)

    """
    plt.figure(figsize=(6, 10))

    plt.subplot(2, 1, 1)
    image = process_image(image_path)
    plt.imshow(image)

    plt.subplot(2, 1, 2)
    plt.barh(top_classes, top_probs, align = 'center')
    plt.yticks(top_classes)
    plt.xlabel('Probability')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    """

if __name__ == "__main__":
    main()