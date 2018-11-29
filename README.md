# imagenet-transfer-learning
Using Keras/tensorflow and Pytorch to retrain image classification models trained on Imagenet for a binary image classification task

## Usage
Each .py file (with the exception of split.py) is a different model. They are named for the pretrained model that they use mnet (= mobilenet), nasnet, or vgg(19). They expect to find data in a folder name "data" containing subdirectories "train" and "test" for the corresponding splits. Each of those two subdirectories should contain two folders, each named for one of the two classes of images to be classified, and containing images of those classes. The split.py file splits images sorted by class, but not sorted by train / test, into train and test splits.

Data have not been provided here due to the size of image data. However, my dataset contained images of spiders and scorpions, and mobilenet achieved a score of 94% distinguishing the two.
