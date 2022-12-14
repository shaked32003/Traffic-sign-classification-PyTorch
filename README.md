# Traffic-sign-classification-with-Pytorch
# introduction
An independent project that is not related to academic studies in which I performed a classification for different traffic signs using the pytorch library and **gat 99.87% accuracy in the training phase and 94.25% accuracy in the test phase**
Here I present a quick overview of the project.

![D9456E33-A739-42A4-AE25-761EAF5E5BA7_4_5005_c](https://user-images.githubusercontent.com/96596252/193057664-1bd70f31-2f14-48dc-a8d3-8990ab539d7c.jpeg)

# dataset
The data included 43 different classes where each class represents a different road sign
and taken from the Kaggle site (https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?datasetId=82373&searchQuery=PyTorch)
The size of the data was about 40k images from the various departments
And as part of the data analysis and processing it was possible to see that the same images appear a large number of times in all the different departments
Therefore I decided to implemented data augmentation techniques for the sake of expanding the variety and additional regulation techniques in the network that will be presented later. all this in addition to cataloged, normalized and created a convenient dataframe for the test phase


# model
From a general look at the data it was evident that there is no need to implement a model capable of detecting small details in the image since it can be seen that the main part of the image is the sign of movement and already in the first convolution layers it was possible to achieve a relatively clear clarification of the model therefore I chose to use the ResNet18 architecture as well as one of the main considerations What I was considering was whether to perform pri-trining for the model or not when in the end I chose that the model must be pri-trined (in the trining phase I will expand on the process)

The loss function, learning rate, and optimizer were the same for all models:
- loss function: CrossEntropyLoss
- optimizer: Adam
- lr: 0.0001

# Training
First I trained the model without pre-training, although I assumed that there was an overlap between the domains, it would be useful to use the prior knowledge of the network for learning, but it was important to test the effect without pre-training first
I examined the result for 25 epoch and got the following results:

 <img width="750" alt="?????????? ?????? 2022-09-19 ??-0 20 22" src="https://user-images.githubusercontent.com/96596252/190928588-a7855ef8-9078-40fe-b150-b77e78247616.png">
 
As you can see it is clear that the pre-training contributes but as you could see the accuracy levels fluctuated in the 65 percent area as a limiter to the fact that the loss value had difficulty falling below the value of 1 and fluctuated around it

The optimizer I started with was adam
I decided to try SGD with a batch size of 32 and LR=0.001 when I made a gradual lowering of it starting from the 7th iteration and the results I got:

<img width="759" alt="?????????? ?????? 2022-09-19 ??-0 20 39" src="https://user-images.githubusercontent.com/96596252/190928585-cd25a21d-8858-44a3-825e-a4d008db5cb9.png">
 
Results on the face of it were almost the same and with an easy win for Adam I decided to stay with him
It was also possible to notice that the loss problem I mentioned above still exists

After trying different regularization methods in the optimizer, I assumed that there might be a need for regularization methods in the network itself, such as dropuot
By using the feature extraction technique, I made another change in the last FC layer of the network beyond its initial adaptation to my classification problem, it is worth improving the loss value and the accuracy (you can see the changes I made in the code notebook)

The result was a decrease in the value of the loss, which did reach the 0.523 area, but the accuracy percentage remained the same around 67%

Now I assumed that a fine tuning technique should be performed on some of the last layers of the convolution. after examining the amount of layers on which the fine tuning technique should be performed
As a limiter for lowering the value of lr = 0.0001 in order to get small fluctuations in the weights, I got excellent results for fine tuning for the last 2 convolution layers
with an train accuracy percentage of 94.21%,test accuracy of 90.43 and a zero loss value

<img width="706" alt="?????????? ?????? 2022-09-25 ??-1 04 18" src="https://user-images.githubusercontent.com/96596252/192120243-4a9ed827-9679-4306-bb04-e591a3ad2628.png">

# evaluation
Following the last graph presented
Although the model did not go into overfit, but
I saw that the model had difficulty rising from the 94% area in the training phase, and likewise in direct relation it stopped in the 90% area in the test phase

My goal at the moment was to check whether it is possible to improve the model's performance without entering a state of overfit:
For this purpose, I decided to download a certain augmentation method, but from the considerations mentioned in the dataset part, it was important not to download all of them, so I left `transforms.RandomHorizontalFlip()`

After that, I ran the model for additional training with a larger amount of epoch (35) in order to check if there is a breaking point where the model will go into overfit

The results were impressive with 99.87% accuracy in the training phase and 94.25% accuracy in the test phase

<img width="711" alt="?????????? ?????? 2022-09-25 ??-12 05 53" src="https://user-images.githubusercontent.com/96596252/192136589-eb9650d6-7eb3-4a7c-b9cd-92cfd1e36b1f.png">

# conclusions
In general, it can be concluded that the pre-training technique along with the various transfer-learning methods mentioned helped to save valuable training time along with good performance
Along with the regularization methods and relatively large data that allowed the model not to reach overfitting for the training data
