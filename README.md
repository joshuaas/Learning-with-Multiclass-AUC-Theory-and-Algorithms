# Learning with Multiclass AUC: Theory and Algorithms 
> Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao and Qingming Huang. [Learning with Multiclass AUC: Theory and Algorithms](). IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021. (Regular Paper)


# AUC_Traditional

## Note
This implementation is inspired by scikit-learn, and we extended it to our work (especially, the special cv regime).

- numpy
- scikit-learn
- pandas
- pickle

## Acceleration
[Cython](https://cython.org/) is an efficient extension for python to accelerate the code.

We also accelerate this code with cython and achieve remarable performance!(Refer to cfunc.pyx and setup.py)

To execute experiments with cython, u should compile the cython code first .

```
python3 setup.py build_ext --inplace
cython -a cfunc.pyx 
```

Then, set the bool cython=True, to enjoy the acceleration.
## Run
When u have already set the corresponding parameters, then
```
python3 train.py
```





