# Generative Adversarial Networks  

Generative Adversarial Networks를 구현했습니다.   

---

### Pytorch   
- 7.1 : MNIST 손글씨를 생성하는 GAN 모델입니다.  
- 7.2, 7.3 : MNIST 손글씨를 생성하는 DCGAN 모델입니다. 7.2는 DataLoader를 사용하지 않고 Data를 Load했으며, 7.3은 DataLoader를 사용했습니다. 
- 7.4 : 개와 고양이를 생성하는 DCGAN모델입니다. 그러나 Dataset의 편차가 매우 심해 좋은 성능이 나오지 않습니다.
- 7.5 : 오직 고양이의 얼굴 부분만 있는 Dataset으로 학습시켜, 고양이의 얼굴을 생성하는 DCGAN 모델입니다.   


### Tensorflow   
- 7.6 : MNIST 손글씨를 생성하는 GAN 모델입니다. 
- 7.7 : MNIST 손글씨를 생성하는 DCGAN 모델입니다.

---



## Code  

- [7.1 GAN, Dataset : MNIST](7_GAN/7.1_GAN_MNIST_Pytorch.py)  

- [7.2 DCGAN - No DataLoader, Dataset : MNIST](7_GAN/7.2_DCGAN_MNIST_No_DataLoader_Pytorch.py)  

- [7.3 DCGAN - Use DataLoader, Dataset : MNIST](7_GAN/7.3_DCGAN_MNIST_Use_DataLoader_Pytorch.py)  

- [7.4 DCGAN, Dataset : Dogs vs Cats](7_GAN/7.4_DCGAN_dogs_vs_cats_Pytorch.py)  

- [7.5 DCGAN, Dataset : Cats](7_GAN/7.5_DCGAN_cats_Pytorch.py)  


---   

- [7.6 GAN, Dataset : MNIST](7_GAN/7.6_GAN_MNIST_Tensorflow.py)  

- [7.7 DCGAN, Dataset : Cats](7_GAN/7.7_DCGAN_MNIST_Tensorflow.py)    


## Datasets  

- [MNIST.csv Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  

- [Dogs vs Cats Dataset](https://www.kaggle.com/competitions/dogs-vs-cats/data)  

- [Cats Dataset](https://www.kaggle.com/datasets/veeralakrishna/cat-faces-dataset?resource=download-directory)
