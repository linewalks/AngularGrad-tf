# AngularGrad-tf

Tensorflow implementation of [AngularGrad: A New Optimization Technique for Angular Convergence of Convolutional Neural Networks](http://arxiv.org/abs/2105.10190).

The official implementation of AngularGrad is [mhaut/AngularGrad](https://github.com/mhaut/AngularGrad).

## How to use

You can import the optimizer as follows:

### AngularGrad(cos)
```python
from angular_grad import AngularGrad
...
model = YourModel()
...
model.compile(optimizer=AngularGrad("cos"), ...)
...
```

Or you can omit a value "cos".
```python
...
model.compile(optimizer=AngularGrad(), ...)
...
```

### AngularGrad(tan)
```python
from angular_grad import AngularGrad
...
model = YourModel()
...
model.compile(optimizer=AngularGrad("tan"), ...)
...
```


## Params
```python
AngularGrad(
    method_angle: str = "cos",
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.999,
    eps=1e-7,
    name: str = "AngularGrad",
    **kwargs
)
```

## Tested version
- Python 3.6.9
- Tensorflow 2.5.0

Developed by Eunchan Lee(eunchan@linewalks.com), 2021 [Linewalks](linewalks.com).

If there is any problem in this repository, please feel free to contact us at the above email address.
