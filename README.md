# MNIST digit generation with GAN

Generation of MNIST digits using a Generative Adverserial Network (GAN).

<p align="center">
    <img src="mnist-gan.gif" height="440px">
</p>

## Install dependencies

Create a Python 3 virtual environment and activate it:

```bash
virtualenv -p python3 venv
source ./venv/bin/activate
```

Next, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Train model

Start the model training by running:

```bash
python gan.py
```

To track metrics, start `Tensorboard`

```bash
tensorboard --logdir path/to/log/dir
```

and then go to [localhost:6006](localhost:6006).
