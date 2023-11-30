<H1> Demo Fashion-MNIST (IML4E) </H1>

Demo MLOps project for the IML4E OSS experimentation platform.

![](tutorials/img/iml4e_full.png)

[TOC]

## Project structure

- [`conf/`](conf): Configuration files.
- [`training/`](training): Source code, including data processing, models and training code.
- [`pipeline/`](pipeline): Kubeflow components and pipeline for workflow orchestration in a kubernetes cluster.
- [`build.sh`](build.sh): Script to build and push the docker image.
- [`tests/`](tests): Pytest unit-tests.
- [`tutorials/`](tutorials): Tutorials for using the workflow and learning about the MLOps component.

## Setup

Install base dependencies:

```bash
pip install --upgrade pip
pip install -e '.[tf,kfp]'
```

The option `tf` above installs TensorFlow. Do not include this if you install TensorFlow using [some other method](https://www.tensorflow.org/install).

### Infrastructure setup

All training scripts can be run locally. However, a kubernetes cluster with the
following resources is needed in order to run the Kubeflow pipeline:

* [kubeflow pipelines](https://www.kubeflow.org/docs/components/pipelines/introduction/)
* [MLFlow](https://mlflow.org/) server
* Image registry

This demo uses the [OSS experimentation platform](https://gitlab.fokus.fraunhofer.de/iml4e/iml4e_oss_exp_platform/-/tree/main).
Clone the OSS experimentation platform [repository](https://gitlab.fokus.fraunhofer.de/iml4e/iml4e_oss_exp_platform/-/tree/main)
and follow the instructions to install the platform locally.

> **Note**: When installing the OSS experimentation platform, remember to install the optional local image registry, as it is required for this demo.

## Configuration files

Most configurable settings are defined and loaded from external [Hydra](https://hydra.cc/)
yaml files in the [`conf/`](conf) directory.

## Training

The [`training`](training) directory contains all the script needed for the training
pipeline. The main steps of the training workflow are:

### 1. Pull data

The [`pull_data.py`](training/pull_data.py) script is used to pull the data tracked
by DVC.

```bash
python -m training.pull_data
```
For more help, see `python -m training.pull_data --help`.

### 2. Preprocessing data

Script to preprocess the data ([`preprocess_data.py`](training/preprocess_data.py)). 
Code for data processing should be added here.

```bash
python -m training.preprocess_data
```
For more details, see `python -m training.preprocess_data --help`.

### 3. Training

Define your models in the [`model.py`](training/model.py) script and add its
corresponding build functions to the `MODELS_FACTORY`. Add here the implementation
details needed to interact with the models (train, evaluate, save, etc.).

Training and model parameters are loaded from a training config file in the
[`conf/`](./conf) directory. If you need to add or modify configuration variables,
add/modify the configs in [`conf/training`](conf/training) and the values are
automatically available in the loaded configuration object in the code.

Run the training with the [`train.py`](training/train.py) script.

```bash
python -m training.train my-experiment-name
```
For more details, see `python -m training.train --help`.

The [`tracker.py`](./training/tracker.py) script contains the code needed for model and
experiment tracking.

There are three ways to track the run: storing data to filesystem,
using local containerized tracking service, and using an external tracking service.
Either way, MLFLow's library is used to track the parameters, metadata, and artifacts
including the model. The model training output is written to a folder specified in the
input arguments.

To learn more on How to define and train your model, check out the [2-Build-and-train-your-own-model.md](tutorials/2-Build-and-train-your-own-model.md) tutorial.

### 4. Evaluation

The [`evaluate.py`](./training/evaluate.py) script is used to read and compare the
evaluation metrics generated during training against the thresholds values defined in the 
[`threshold_metrics_for_evaluation.json`](./conf/threshold_metrics_for_evaluation.json)

```bash
python -m training.evaluate
```
For more details, see `python -m training.evaluate --help`.

To learn more on how the evaluation works, check out the [3-Model-evaluation.md](tutorials/3-Model-evaluation.md) tutorial.

### 5. Registration

Finally, the [`register`](./training/register.py) script is used to register a trained
model if it passes the evaluation criteria.

```bash
python -m training.register my-model-registry-name
```
For more details, see `python -m training.register --help`.

## Pipeline

The [`pipeline`](./pipeline) directory contains the Kubeflow pipeline used to
orchestrate the training workflow. 

The above steps are used here as components of the pipeline:

- [`pull_data_component.yaml`](./pipeline/components/pull_data_component.yaml)
- [`preprocess_data_component.yaml`](./pipeline/components/preprocess_data_component.yaml)
- [`train_component.yaml`](./pipeline/components/train_component.yaml)
- [`evaluation_component.yaml`](./pipeline/components/evaluation_component.yaml)
- [`registration_component.yaml`](./pipeline/components/registration_component.yaml)

Pipeline related settings are also loaded from config files in the
[`conf/`](conf) directory.

To learn more on how to build a kubeflow pipelines, check out the [1-Model-training-pipeline-overview.md](tutorials/1-Model-training-pipeline-overview.md) tutorial.

The steps to run the training pipeline remotely in a cluster using kubeflow pipelines are:

### 1. Build and push the Docker image (if needed)

Use the [`build.sh`](build.sh) script to build and push the training image to the
image registry.

```bash
export IMAGE_TAG=t001

./build.sh -p $IMAGE_TAG
```
For more details, see `./build.sh -h`.

> **Note**: It is set up to use a local image registry (local setup).
> If you want to use an external registry, you need to modify the image repository address
> in the [build script](build.sh) and [pipeline config](conf/pipeline/compile_config/default.yaml).

### 2. Compile pipeline to an Argo Workflow yaml

Compile the Kubeflow Pipeline with the newly created Docker image tag.

```bash
python -m pipeline.compile $IMAGE_TAG [--output pipeline.yaml]
```
For more details, see `python -m pipeline.compile --help`.

### 3. Submit the pipeline

First, port-forward the Kubeflow ingress gateway:

```shell
kubectl port-forward --namespace istio-system svc/istio-ingressgateway 8080:80
```

Now Kubeflow pipelines UI should be reachable at [http://localhost:8080](http://localhost:8080/)

> User / Password: user@example.com / 12341234

The [`submit.py`](pipeline/submit.py) script is used to submit the compiled pipeline to
the Kubeflow cluster.

```bash
python -m pipeline.submit --register --kubeflow-url http://localhost:8080 --kubeflow-username user@example.com --kubeflow-password 12341234 --namespace kubeflow-user-example-com
```
For more details, see `python -m pipeline.submit --help`.

If no host URI is specified, it will use the cluster
currently configured in the [`kubectl`](https://kubernetes.io/docs/tasks/tools/) `current-context`.

The parameters present in the config yamls can also be overridden through the CLI. For example,
to override the value of `submit_config.experiment: default` and `submit_config.run: `,
you can pass them as

```bash
python -m pipeline.submit submit_config.experiment=myExperimentName submit_config.run=myRunName
```

## Access Kubeflow

Access the Kubeflow pipelines dashboard using `kubectl port-forward`:

```bash
kubectl port-forward --namespace kubeflow svc/ml-pipeline-ui 8080:80
```
Now Kubeflow pipelines UI should be reachable at [http://localhost:8080](http://localhost:8080/)

> User / Password: user@example.com / 12341234

## Access MLflow

### a. Using local filesystem for tracking

If you are using the local file system for tracking, or just running the training locally
without setting up any MLFlow server, MLflow will create a `mlflow.db` file and a `mlruns/`
directory where the logs, model and all data about it will be stored.

You can access MLflow Tracking UI by running the following command on
the directory containing the newly created `mlruns/` and `mlflow.db` artifacts

```bash
mlflow ui
```
and viewing it at [http://localhost:5000](http://localhost:5000).

> If you're using folders for tracking and there are errors about missing meta.yaml
> file, try removing mlruns/ directory and trying again.

### b. Using external MLflow server

If you are using a remote MLflow server in a kubernetes cluster, the MLflow UI should by
accessible by using `kubectl port-forward`:

```bash
$ kubectl port-forward svc/mlflow 3000:5000 --namespace mlflow
```
Now MLFlow's UI should be reachable at [http://localhost:3000](http://localhost:3000).


## How to test

The project is tested with pytest. Run tests with:

```bash
# make sure you install the "dev" requirements
pip install -e '.[tf,kfp,dev]'

# run tests
pytest
```