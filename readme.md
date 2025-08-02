# Apple Disease Classification (by leaves) using VGG16

---

## Project Pipeline
---
- Dataset downloading from :  https://www.kaggle.com/datasets/ludehsar/apple-disease-dataset
- Dataset contain 8161 Images of leaves, classified into Train and Test
- Each having 4 classes of Images : **Scab Apples, Black Rot Apple, Cedar Apple and Healthy Apple**
### Training Dataset - 6218
---
- **Scab Apples** - 2016 images
- **Black Rot Apples** - 1987 images
- **Cedar Apples** - 1760 images
- **Healthy Apples** - 2008 images
### Validation Dataset - 20% of Training Dataset - 1553 images
### Test Dataset - 1943
---
- **Scab Apples** - 504 images
- **Black Rot Apples** - 497 images
- **Cedar Apples** - 440 images
- **Healthy Apples** - 502 images

- After splitting the dataset we can see it is unbalanced, so we balance the data with the help of class weights.
### Pre Training
---
- Creating new sequential model
- Adding `VGG16` as our model for pre training
- Normalizing activations from `VGG16` - helps in speed up training
- Downsampling the feature maps - helps in reducing the chances of overfitting
- Flatten - to change the image from 2D to 1D vector
-