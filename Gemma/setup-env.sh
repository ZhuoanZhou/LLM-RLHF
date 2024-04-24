
python=3.11
env_name=trl-gemma
mamba create --yes -p /home/jovyan/${env_name} python=${python} ipykernel

conda init
source ~/.bashrc
conda activate /home/jovyan/${env_name}
python -m ipykernel install --user --name=${env_name}-kernel


# Update this to whatever mlflow version your cluster has currently
pip install mlflow==2.2.2

pip install "torch==2.1.2" tensorboard

pip install  --upgrade \
    "transformers==4.38.2" \
    "datasets==2.16.1" \
    "accelerate==0.26.1" \
    "evaluate==0.4.1" \
    "bitsandbytes==0.42.0" \
    "trl==0.7.11" \
    "peft==0.8.2"


pip install ninja packaging
pip install flash-attn --no-build-isolation --upgrade


