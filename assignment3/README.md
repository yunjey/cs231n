In this assignment you will implement recurrent networks, and apply them to image captioning on Microsoft COCO. We will also introduce the TinyImageNet dataset, and use a pretrained model on this dataset to explore different applications of image gradients.

The goals of this assignment are as follows:

- understand the architecture of **recurrent neural networks (RNNs)** and how they operate on sequences by sharing weights over time
- understand the difference between vanilla RNNs and **Long-Short Term Memory (LSTM)**
- understand how to **sample from an RNN at test-time**
- understand how to combine convolutional neural nets and recurrent nets to implement an **image captioning system**
- understand how a trained convolutional network can be used to **compute gradients** with respect to the input image
- implement and different applications of image gradients, including **saliency maps**, **fooling images**, **class visualizations**, **feature inversion**, and **DeepDream**.

</br>

### Q1: Image Captioning with Vanilla RNNs (Completed)

The IPython notebook `RNN_Captioning.ipynb` will walk you through the implementation of an image captioning system on MS-COCO using vanilla recurrent networks.

### Q2: Image Captioning with LSTMs (Completed)

The IPython notebook `LSTM_Captioning.ipynb` will walk you through the implementation of Long-Short Term Memory (LSTM) RNNs, and apply them to image captioning on MS-COCO.

### Q3: Image Gradients: Saliency maps and Fooling Images (Not Yet)

The IPython notebook `ImageGradients.ipynb` will introduce the TinyImageNet dataset. You will use a pretrained model on this dataset to compute gradients with respect to the image, and use them to produce saliency maps and fooling images.

### Q4: Image Generation: Classes, Inversion, DeepDream (Not Yet)

In the IPython notebook `ImageGeneration.ipynb` you will use the pretrained TinyImageNet model to generate images. In particular you will generate class visualizations and implement feature inversion and DeepDream.



