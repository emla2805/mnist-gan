# MNIST digit generation with GAN

Generate MNIST digits with Generative Adverserial Network (GAN).


## Install dependencies

Development for this example will be isolated in a Python virtual environment.
This allows us to experiment with different versions of dependencies.

There are many ways to install `virtualenv`, see the
[TensorFlow install guides](https://www.tensorflow.org/install) for different
platforms, but here are a couple:

* For Linux:

        sudo apt-get install python-pip python-virtualenv python-dev build-essential

* For Mac:

        pip install --upgrade virtualenv

Create a Python 3.7 virtual environment for this example and activate the
`virtualenv`:

    virtualenv -p python3.7 venv
    source ./venv/bin/activate

Next, install the required dependencies:

    pip install -r requirements.txt

## Train model

Start the model training by running:

    python gan.py

To track metrics, start `Tensorboard`

    tensorboard --logdir <LOGDIR>

and navigate to [localhost:6006](localhost:6006).
