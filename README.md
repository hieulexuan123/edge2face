# Edge2face

Edge2face is a personal project that utilizes the [Pix2pix GAN](https://arxiv.org/abs/1611.07004), implemented in PyTorch, to transform simple hand-drawn faces into realistic faces.

## Dataset

The image pairs were created using the [**CelebAMask-HQ dataset**](https://github.com/switchablenorms/CelebAMask-HQ). The script for creating these pairs can be found in `data/create_dataset.py`. The dataset, consisting of nearly 4000 images, which I used to train the model, is available on [Kaggle](https://www.kaggle.com/datasets/lexuanhieuai/edge2face-dataset/).

## Pretrained Model

You can download the pretrained model from my Hugging Face space.

## Demo

Try out the model on [Hugging Face Spaces](https://huggingface.co/spaces/HieuLeXuan/Edge2face).

Feel free to explore and experiment with the model!
