# Factorized-Spatial-Embeddings

Tensorflow implementation of [Unsupervised learning of object landmarks by factorized spatial embeddings](http://www.robots.ox.ac.uk/~vedaldi//assets/pubs/thewlis17unsupervised.pdf) by Thewlis el al. for unsupervised landmark detection. 

### Sample results
Test results on LFW with 8 landmarks (K=8, M=4), trained on CelebA dataset for 2 epochs.
![](https://github.com/alldbi/Factorized-Spatial-Embeddings/blob/master/test_samples/test-K8M4.png)
Test results on LFW with 16 landmarks (K=16, M=4), trained on CelebA dataset for 2 epochs.
![](https://github.com/alldbi/Factorized-Spatial-Embeddings/blob/master/test_samples/K16M4.png)
## Setup

### Prerequisites
- Tensorflow 1.4

### Getting Started
First download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or the [UT Zappos50k shoes dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/), extract images and use them to train the model.  
```sh
# clone this repo
https://github.com/alldbi/Factorized-Spatial-Embeddings.git
cd Factorized-Spatial-Embeddings
# train the model 
python main.py \
  --mode train \
  --input_dir (directory containing CelebA dataset) \ 
  --K 8  \ #number of landmarks to be learned

# test the model
python main.py \
  --mode test \
  --input_dir (directory containing testing images)
  --checkpoint (address of the trained model, which is /OUTPUT as default)
  --K 8
  


