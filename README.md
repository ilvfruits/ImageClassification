# ImageClassification
Files to support Udacity Image Classification Project submission

**Create Your Own Image Image Classifier**

by Xin Yang

**Introduction:**
This project aim to implement an image classifier with PyTorch.

**Contents**
1. README.md
2. ImageClassifierProject_Part1_XinYang.html - Jupyter Notebook from Part 1 development
3. train.py - Python code to train an image classifier model
4. predict.py - Python code to predict image class using this model
5. checkpoint.pth - the model training data (epoch = 3, accuracy > 75%)  
'checkpoin.pth' is too big to upload here. Link to google driver: https://drive.google.com/file/d/1hRnubedNeFM-AzGZogWKWeq2Z1thUAKy/view?usp=drive_link
7. training_log.txt - terminal outputs for model training
8. ImageClassifierProject_Part1.html - the html file for the Jupyter 

**Explaination**
1. train.py takes one argument for data directory
e.g. % python train.py flowers

2. train.py takes optional arguments  
--save_dir  
--arch (default: vgg16): allows users to choose architectures available from torchvision.models  
--learning_rate (default: 0.001): the learning rate for model training  
--hidden_units (default: 512): the number of hidden units  
--epochs (default: 3): training epochs for the model  
--gpu: allows users to choose training the model on a GPU  
e.g. % python train.py flowers --arch vgg16 --learning_rate 0.01 --hidden_units 512 --epochs 10 --gpu  

3. predict.py takes two arguments for path to the image and model checkpoint. It required GPU support to evaluate the model because the model (checkpoint.pth) was trained on GPU.  
e.g. % python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --gpu

4. predict.py takes optional arguments  
--topk (default: 1): number of top classes  
--mapfile (default: None): mapping file for the category names  
--gpu: allows users to choose training the model on a GPU  
e.g. % python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --topk 5 --mapfile cat_to_name.json --gpu  

**Supporting links**
Screen recording for running train.py and predict.py are uploaded to YouTube.
1. Screen recording for train.py running on GPU of Udacity workspace.  
epoch 1 completed at 6'43" with an accuracy of 0.764.  
epoch 2 completed at 13'40" with an accuracy of 0.840.  
epoch 3 completed at 13'31" with an accuracy of 0.869.  
https://youtu.be/Rnxj8hlc8do

2. Screen recording for predict.py running on GPU of Udacity workspace. Results can only be printed to the terminal due to the error message of "Could not connect to display: 1".
https://youtu.be/5RedXvY8umQ

3. Screen recording for predict_test.py running on local CPU for a simplified model (to demo the function of displaying images). The checkpoint.pth is also trained on local CPU with only 2 categories of flowers. The images and histograms can be displayed.
https://youtu.be/Xaw6PO6c730
