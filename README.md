# Factorized-Spatial-Embeddings

Tensorflow implementation of [Unsupervised learning of object landmarks by factorized spatial embeddings](http://www.robots.ox.ac.uk/~vedaldi//assets/pubs/thewlis17unsupervised.pdf) by Thewlis el al. 

### Sample results
Test results on LFW with 8 landmarks (K=8, M=4) 
![](https://github.com/alldbi/Factorized-Spatial-Embeddings/blob/master/test_samples/test-K8M4.png)

## Setup

### Prerequisites
- Tensorflow 1.4

### Getting Started

```sh
# clone this repo
https://github.com/alldbi/Factorized-Spatial-Embeddings.git
cd Factorized-Spatial-Embeddings
# train the model 
python main.py \
  --mode train \
  --input_dir (directory containing CelebA dataset)

# test the model
python main.py \
  --mode test \
  --input_dir (directory containing testing images)


