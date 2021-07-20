### IUPAC2Struct model 

This is Transformer-based IUPAC2Struct model. It converts IUPAC names to SMILES strings. The model's quality is almost on the same level as rules-based OPSIN; however, our solution is purely neural one.  

| Model      | Accuracy |
| ----------- | ----------- |
| OPSIN       | 99.4%       |
| IUPAC2Struct  | 99.1%        | 

## Usage
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

## Citation
Please, cite: 

_Krasnov, L., Khokhlov, I., Fedorov, M.V., Sosnin, S. Transformer-based artificial neural networks for the conversion between chemical notations. Sci Rep 11, 14798 (2021). https://doi.org/10.1038/s41598-021-94082-y_

````
@article{Krasnov2021,
  doi = {10.1038/s41598-021-94082-y},
  url = {https://doi.org/10.1038/s41598-021-94082-y},
  year = {2021},
  month = jul,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {11},
  number = {1},
  author = {Lev Krasnov and Ivan Khokhlov and Maxim V. Fedorov and Sergey Sosnin},
  title = {Transformer-based artificial neural networks for the conversion between chemical notations},
  journal = {Scientific Reports}
}
