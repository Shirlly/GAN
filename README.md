# AuGAN

An implementation of AuGan to generate images (MNIST, Cifar10, Cifar100)


to train simply run
```bash
python main.py --train --modelclass AuGan
```

We add the generated images to the original data to see whether there is improvement for data classification.


## Required packages:
* Python 2.7
* Tensorgraph (https://pypi.python.org/pypi/tensorgraph)
* Tensorflow 1.0
