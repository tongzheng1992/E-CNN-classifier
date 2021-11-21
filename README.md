# E-CNN-classifier
This is the avaiable code for the paper ["An evidential classifier based on Dempster-Shafer theory and deep learning"](https://arxiv.org/abs/2103.13549) (arXiv:2103.13549).

Codes for Dempster-Shafer layer and utility layer are in the file "libs", as well as the metrics "average utility".

The file "demo.ipynb" provides a demo about how to build, train, and interfere precise and imprecise classification with an evidential CNN classifier with the Dempster-Shafer layer and utility layers.

The file "weights_zoo" includes the parameters of a trained evidential CNN classifier that are used in the demo.

The required libraries and their version:

python == 3.7.10

tensorflow == 2.4.1.

-------------------------
#Update on Nov. 21th, 2021

Thanks to [paul-bd](https://github.com/paul-bd) for his/her reimplementation of Demspter-Shafer layer in the framework of Pytorch, see https://github.com/paul-bd/DempsterShaferTheory
