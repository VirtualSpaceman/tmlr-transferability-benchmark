#!/bin/bash

eval "$(conda shell.bash hook)"

if ! conda activate stan; then
    conda create -y -n stan python=3.10
    conda activate stan
fi
if conda activate stan; then
    python -m pip install -U pip
    python -m pip install arviz matplotlib numpy pandas pystan scikit-learn scipy
else
    echo "ERROR: could not create or activate conda environment 'stan'"
fi
