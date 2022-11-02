<h1> Building and training your custom model </h1>

This tutorial will guide you on how to create and train your custom model.

[TOC]

## Concepts and terminology

- **Model definition**: The model src code is defined in [`training/model.py`](/training/model.py). Also in this file, we can define model and training parameters.
- **Model training**: The model training src code is defined in [`training/train.py`](/training/train.py) which requires the model to be trained and the parameters needed to train the model. Model training is exposed as a CLI script to be used as a component in the training pipeline.
- **Configuration**: Configuration is loaded from YAML files located in [`/conf`](/conf). This project uses [Hydra](https://hydra.cc/docs/intro/) for configuration management.

## Define model

> If you need to add or modify configuration variables, modify the [`conf/training/`](/conf/training/) files and the values are automatically available in the loaded configuration object in the code.

There is an existing demo model which can be trained without any modification to the repository.

You can change the model by:

#### Scenario 1 (Modification of the demo model)

Modify the existing model in [`training/model.py`](/training/model.py). Make the necessary adjustments to model compilation and training parameters.

#### Scenario 2 (Define a new model)

Create the model compilation function in [`training/model.py`](/src/training/model.py) and register the model to be available for selection in [`training/train.py`](/training/train.py).

Modify the train CLI script in [`training/train.py`](training/train.py) to take into account the new model and its parameters. Change the default arguments accordingly.

## Unit test

Run the unit test to make sure the updated code syntax is correct and no bug is introduced. Modify the tests if needed. To run the unit test run the following command in your virtual environment.

```bash
pip install --upgrade pip
# install dev dependency
pip install -e '.[tf,kfp,dev]'
# run pytest
pytest
```
