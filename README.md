# Traffic-sign-classification-with-Pytorch

A self-study project that is not related to academic studies in which I performed a classification for different traffic signs using the pytorch library
Here I present a quick overview of the project. For a complete walkthrough, including the code, please head to my notebook

# dataset:
The data included 43 different classes where each class represents a different road sign
and taken from the Kaggle site (_)
The size of the data was about 40k images from the various departments
And as part of the data analysis and processing, I implemented data augmentation techniques, cataloged, normalized and created a convenient dataframe for the test phase


# model:
From a general look at the data it was evident that there is no need to implement a model capable of detecting small details in the image since it can be seen that the main part of the image is the sign of movement and already in the first convolution layers it was possible to achieve a relatively clear clarification of the model therefore I chose to use the ResNet18 architecture as well as one of the main considerations What I was considering was whether to perform pri-trining for the model or not when in the end I chose that the model must be pri-trined (in the trining phase I will expand on the process)

The loss function, learning rate, and optimizer were the same for all models:
- loss function: CrossEntropyLoss
- optimizer: Adam
-lr: 0.0001

![Original-ResNet-18-Architecture](https://user-images.githubusercontent.com/96596252/190928402-dc64770d-6cd2-447a-9e3a-2d90519fd84e.png)

# Training:
First I trained the model without pre-training, although I assumed that there was an overlap between the domains, it would be useful to use the prior knowledge of the network for learning, but it was important to test the effect without pre-training first
I examined the result for 25 epoch and got the following results:

<img width="1163" alt="צילום מסך 2022-09-18 ב-21 00 25" src="https://user-images.githubusercontent.com/96596252/190928321-23dc8501-fcc2-4202-9fa6-51452cff4f7a.png">
 
As you can see it is clear that the pre-training contributes but as you could see the accuracy levels fluctuated in the 65 percent area as a limiter to the fact that the loss value had difficulty falling below the value of 1 and fluctuated around it

The optimizer I started with was adam
I decided to try SGD with a batch size of 32 and LR=0.001 when I made a gradual lowering of it starting from the 7th iteration and the results I got:

<img width="1163" alt="צילום מסך 2022-09-18 ב-21 00 25" src="https://user-images.githubusercontent.com/96596252/190928335-0d8f3bb5-b091-4bcf-be31-86a86ab49a65.png">
 
Results on the face of it were almost the same and with an easy win for Adam I decided to stay with him
It was also possible to notice that the loss problem I mentioned above still exists

After trying to administer different regularization methods on the optimizer, I assumed that there might be a need for regularization methods in the network itself, such as dropuot
I decided to make another change to the last FC layer of the network beyond its initial adaptation to my classification problem, it is worth improving the loss value and the accuracy which can be seen in the code notebook
