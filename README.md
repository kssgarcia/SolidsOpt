
The code is written in Python and it depends on [Numpy](https://numpy.org), [scipy](https://scipy.org), [TensorFlow](https://www.tensorflow.org) and [SolidsPy](https://solidspy.readthedocs.io/en/latest/).


### Installation

```sh
pip install solidsopt
```

### Load weights for neural networks

```sh
load model
```

## Repositories 

Two repositories were created from this project, one contains the structural optimization algorithms and the other has everything related to the development of deep learning methods.


- [kssgarcia/DeepLearningOpt](https://github.com/kssgarcia/DeepLearningOpt)
- [kssgarcia/OptTopolgy](https://github.com/kssgarcia/OptTopolgy)


## Topology optimization repo

- BESO method [BESO.py](https://github.com/kssgarcia/OptTopolgy/blob/main/BESO.py)
- ESO stress based method  [ESO_stress_based.py](https://github.com/kssgarcia/OptTopolgy/blob/main/ESO_stress_based.py)
- ESO stiff based method [ESO_stiff_based.py](https://github.com/kssgarcia/OptTopolgy/blob/main/ESO_stiff_based.py)
- SIMP method [SIMP.py](https://github.com/kssgarcia/OptTopolgy/blob/main/SIMP.py)``

### Instructions

### 1. Clone repository

```sh
git clone https://github.com/kssgarcia/OptTopolgy.git
```

### 2. Download the required packages running the following command

```sh
conda env create -f environment.yml
```

### 3. Install solidspy

```sh
pip install solidspy
```

## Optimization with deep learning repo

- [SIMP_multi.py](https://github.com/kssgarcia/DeepLearningOpt/blob/main/simp/SIMP_multi.py) code used for generate training dataset.
- [CNN.py](https://github.com/kssgarcia/DeepLearningOpt/blob/main/neural_network/CNN.py) code used for training neural network.
- [load_model.py](https://github.com/kssgarcia/DeepLearningOpt/blob/main/neural_network/CNN2.py) code used for load neural network.
- [SIMP_multi_dist.py](https://github.com/kssgarcia/DeepLearningOpt/blob/main/neural_network/SIMP_multi_dist.py) code used for generate dataset with a distributed load.


### Instructions


### 1. Clone repository

```sh
git clone https://github.com/kssgarcia/DeepLearningOpt.git
```

### 2. Download the required packages running the following command

```sh
conda env create -f environment.yml
```

### 3. Install solidspy

```sh
pip install solidspy
```


