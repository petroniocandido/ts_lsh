# ts-lsh - Locality Sensitive Hashing based embedings for High Dimensional Multivariate Time Series

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/pyFTS/)

```mermaid
classDiagram
    LSH : name
    LSH : length
    LSH : hashtype
    LSH : width
    LSH : activation
    LSH : hash()
    LSH : _hashfunction()
    LSH <|-- SignedRandomProjectionLSH
    SignedRandomProjectionLSH : dist
    SignedRandomProjectionLSH : scale
    SignedRandomProjectionLSH : weights
    LSH <|-- MultipleLSH
    MultipleLSH : num
    MultipleLSH : scale
    MultipleLSH : dist
    MultipleLSH : weights
    MultipleLSH <|-- EnsembleLSH
    EnsembleLSH : aggregation
    EnsembleLSH : aggregation_weights
    LSH <|-- StackedLSH
    StackedLSH : layers
    StackedLSH : append()
```
