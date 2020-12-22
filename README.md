###IUPAC2Struct model 
This is our Transformer-based IUPAC2Struct model. 
It converts IUPAC names to SMILES strings. This model's quality is almost on the same level as rules-based OPSIN; however, our solution is purely neural one.  

| Model      | Accuracy |
| ----------- | ----------- |
| OPSIN       | 99.4%       |
| IUPAC2Struct  | 99.1%        | 

##Usage
Create the environment first:

`conda create -f environment.yml`

To run it:
\
`python run.py`

Command line arguments:

``python run.py --help``
```python run.py --help
usage: run.py [-h] [-f FILE] [-r RANDOM] [-b BEAM_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  take a dataset
  -r RANDOM, --random RANDOM
                        take N random molecules from a dataset
  -b BEAM_SIZE, --beam-size BEAM_SIZE
                        Beam size for Transformer
```
Default settings are:

``-f data/test_100000.csv -b 5 -n 1000 ``

##Citation
Please, cite: 

_Krasnov, Lev; Khokhlov, Ivan; Fedorov, Maxim; Sosnin, Sergey (2020): Struct2IUPAC -- Transformer-Based Artificial Neural Network for the Conversion Between Chemical Notations. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.13274732_

````
@article{Krasnov2020,
  doi = {10.26434/chemrxiv.13274732},
  url = {https://doi.org/10.26434/chemrxiv.13274732},
  year = {2020},
  month = nov,
  author = {Lev Krasnov and Ivan Khokhlov and Maxim Fedorov and Sergey Sosnin},
  title = {Struct2IUPAC -- Transformer-Based Artificial Neural Network for the Conversion Between Chemical Notations}
}
