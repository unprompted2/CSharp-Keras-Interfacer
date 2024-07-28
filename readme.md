
### CSharp-Keras-Interface

#### Introduction

- This library offers an interface with Keras, deploying deep learning models in C# applications without needing additional environments.

#### Features

- Quickly extract Keras models
- Easily integrate with existing development cycles
- Fast deployment
- Highly readable code

#### Usage

A [tutorial video](https://www.bilibili.com/video/av93374622) is available for a detailed walk-through. However, you can follow the steps below as well:

1. Place the .H5 model file in the **Input** folder, then run **model_analysis.py** (press **Enter** without any arguments). The weight file will be output in **Output** folder.

2. Place class library files in the root of your program directory.

3. Import **Wineforever.Neuralnet.dll** and reference the namespace:

   ```c#
   using Wineforever.Neuralnet;
   ```


...

Recent updates have optimized network structure and greatly improved forward propagation speed, with a focus solely on forward propagation of the neural network.

#### License

This library follows open-source protocols and is prohibited from being used commercially.

#### Support

If you wish to support the continued development of this project, donations are greatly appreciated via the QR code below, or via bitcoin. Your support is much appreciated.

Bitcoin Wallet:16RpsEY6C1zLZTPZUX8mXK9ozooqhh5YqS

![](http://106.15.93.194/donate/donate.png)