# Generative Adversarial Networks


I have implemented Generative Adversarial Networks using Pytorch.  

(Pytorch를 사용하여 Generative Adversarial Networks를 구현했습니다.)   


- [7.1 GAN, Dataset : MNIST](7_GAN/7.1_GAN_MNIST.py)
7.1 is a GAN model generating MNIST handwritten digits.  
7.1은 MNIST 손글씨를 생성하는 GAN 모델입니다.  

- [7.2 DCGAN - No DataLoader, Dataset : MNIST](7_GAN/7.2_DCGAN_MNIST_No_DataLoader.py)
- [7.3 DCGAN - Use DataLoader, Dataset : MNIST](7_GAN/7.3_DCGAN_MNIST_Use_DataLoader.py)  
7.2 and 7.3 are DCGAN models also for generating MNIST handwritten digits. 7.2 loads the data without using DataLoader, 7.3 Use DataLoader.  
7.2와 7.3은 MNIST 손글씨를 생성하는 DCGAN 모델입니다. 7.2는 DataLoader를 사용하지 않고 Data를 Load했으며, 7.3은 DataLoader를 사용했습니다.  


- [7.4 DCGAN, Dataset : Dogs vs Cats](7_GAN/7.4_DCGAN_dogs_vs_cats.py)
- [7.5 DCGAN - load checkpoint, Dataset : Dogs vs Cats](7_GAN/7.5_DCGAN_dogs_vs_cats_load_ckpt.py)
7.4 is a DCGAN model for generating images of dogs and cats. However, due to significant variance in the dataset, the performance is not optimal.  
7.5 includes the code to load the checkpoint from 7.4.  
7.4는 개와 고양이를 생성하는 DCGAN모델입니다. 그러나 Dataset의 편차가 매우 심해 좋은 성능이 나오지 않습니다.
7.5는 7.4의 checkpoint를 load하는 코드입니다.

- [7.6 DCGAN, Dataset : Cats](7_GAN/7.6_DCGAN_cats.py)  
7.6 is a DCGAN model trained on a dataset consisting only of cat faces, and it generates images of cat faces. 
7.6은 오직 고양이의 얼굴 부분만 있는 Dataset으로 학습시켜, 고양이의 얼굴을 생성하는 DCGAN 모델입니다.


## Datasets  

- [MNIST.csv Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  

- [Naver Movie Review train Dataset](../Datasets/data_naver_movie_ratings_train.txt)
- [Naver Movie Review test Dataset](../Datasets/data_naver_movie_ratings_test.txt)
