# ts-lsh - Locality Sensitive Hashing based embedings for High Dimensional Multivariate Time Series

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/pyFTS/)

```mermaid
classDiagram
LSH <|-- AveryLongClass : Cool
Class03 *-- Class04
Class05 o-- Class06
Class07 .. Class08
Class09 --> C2 : Where am i?
Class09 --* C3
Class09 --|> Class07
Class07 : equals()
Class07 : Object[] elementData
Class01 : size()
Class01 : int chimp
Class01 : int gorilla
Class08 <--> C2: Cool label
```

```mermaid
classDiagram
    LSH: name
    LSH: length
    LSH: hashtype
    LSH: width
```

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
