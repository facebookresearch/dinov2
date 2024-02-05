#!/bin/bash

module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy git

python -m venv .env

source .env/bin/activate

pip install --upgrade pip

pip install -r requirements.txt