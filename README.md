#  Keyword extraction from scientific documents
Spring Semester 2017, University of Trento.

The task and the results are discussed in the [report](report.pdf) file.

## Repository Structure
This repository contains:
 - two folders, `data`, `embeddings` 
 - three main files, `main_crfbs`, `main_crfrs`, `main_crfgen`; 
 - one requirement file `requirements`
 - one .pdf file, `report`; and
 - two .md files, `README` and `LICENCE`.
 
In the `data` folder is contained the dataset used for the evaluation.

In the `embedding` folder is contained a model to try the algorithm.

In the `main_crf*` are contained the three algorithms presented in the `report`.
 
In the `requirements` is contained the pip freeze of the used python environment.
 
In the `report` is contained an overview of the approach and the experimentation with the experimental results.

## Initialize the working environment

Download and install the crfpp framework used.
```bash
% git clone https://github.com/taku910/crfpp.git
% cd crfpp/
% ./configure 
% make
% su
% make install
```
If you do not want the installation of crfpp to occupy common places on your machine please consider setting the prefix flag. Keep in mind that doing so you will need to either set an alias in your command line for the `crf_learn` and and `crf_test` custom location or modify the `crfpp_gen` code. 
```bash
% ./configure --prefix=/somewhere/else/than/usr/local
```

Install the additional python requirements.
```bash
% pip2.7 install -r requirements.txt
```

Finally, please make sure to download the [word_embedding model](https://drive.google.com/file/d/1ovsY-Ld4jWhMLJi70Jdcm2exPBpJaUBi/view?usp=sharing) and replace the given placeholder in [embeddings/lay_512/epo_15/vectors.txt](embeddings/lay_512/epo_15/vectors.txt).

## Running the algorithm 

Running the baseline.
```bash
% python2.7 main_crfbs.py
```

Running the first optimisation with RandomSearch.
```bash
% python2.7 main_crfrs.py
```

Running the second optimisation with Genetic optimiser..
```bash
% python2.7 main_crfgen.py
```

Please note that the default configuration will NOT clean up the sandbox when done.

For any further information about the working of the algorithm do not hesitate to contact me or read the [report.pdf](report.pdf).
