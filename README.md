# RDSKD

This repository includes source code for the paper Pengchao Han, Jihong Park, Shiqiang Wang, and Yejun Liu, 
"Robustness and Diversity Seeking Data-Free Knowledge Distillation," 
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021.


## Getting Started

The code runs on Python 3. To install the dependencies, run
```
pip3 install -r requirements.txt
```

To run RDSKD on the MNIST dataset:

First, train a teacher network, after which the teacher model is saved in the `Model` folder,
```shell
python teacher-train.py
```
And then, train a generator using the teacher,
```shell
python generator-train.py
```
The generator model is saved in the `Model` folder.

Based on the trained teacher and generator models, train the student and save it in the `Model` folder,
```shell
python student-train.py
``` 
The results for training the teacher, generator, and student are saved as CSV files in the `results` folder. 
 The CSV files should be deleted before starting a new round of experiment. Otherwise, the new results will be appended to the existing file.
 
Last, test the models,
```shell
python model_imges_test.py
``` 
The test results is saved in `results\results.csv`.

To run RDSKD on the SVHN dataset:
```shell
python teacher-train.py --dataset SVHN --model WResNet40-2
python generator-train.py --dataset_teacher SVHN --latent_dim 1000 --channels 3
python student-train.py --student_model WResNet16-1 --dataset_teacher SVHN --latent_dim 1000
python model_imges_test.py --dataset_teacher SVHN --latent_dim 1000 --channels 3
```
To run RDSKD on the CIFAR-10 dataset:
```shell
python teacher-train.py --dataset cifar10 --model ResNet34
python generator-train.py --dataset_teacher cifar10 --latent_dim 1000 --channels 3
python student-train.py --student_model ResNet18 --dataset_teacher cifar10 --latent_dim 1000
python model_imges_test.py --dataset_teacher cifar10 --latent_dim 1000 --channels 3
```


### Third-Party Library

Part of this code is adapted from  <https://github.com/bolianchen/Data-Free-Learning-of-Student-Networks>,

<https://github.com/xternalz/WideResNet-pytorch>,

<https://github.com/sbarratt/inception-score-pytorch>,

<https://github.com/mseitzer/pytorch-fid>, 

and <https://github.com/richzhang/PerceptualSimilarity>.

## Citation

When using this code for scientific publications, please kindly cite the above paper.