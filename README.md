# imagenet-transfer-learning
Using Keras/tensorflow and Pytorch to retrain image classification models trained on Imagenet for a binary image classification task

## Usage
Each .py file (with the exception of split.py) is a different model. They are named for the pretrained model that they use mnet (= mobilenet), nasnet, or vgg(19). They expect to find data in a folder name "data" containing subdirectories "train" and "test" for the corresponding splits. Each of those two subdirectories should contain two folders, each named for one of the two classes of images to be classified, and containing images of those classes. The split.py file splits images sorted by class, but not sorted by train / test, into train and test splits.

My dataset consisted of 200 images of spiders and scorpions (100 each). This data cannot be provided on github, due to the size of the dataset and due to copyright issues (feel free to contact me if interested in replicating this small study). However, it is easy to obtain a dataset by scraping google images, and manually cleaning the data to remove images that don't belong. Note that images should be resized to 256x256.

## Results
During testing, the pretrained mobilenet model achieved an accuracy of 94% distinguishing the two classes.
