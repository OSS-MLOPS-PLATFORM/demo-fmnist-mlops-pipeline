# Tutorials

This folder contains tutorials to get you started with building and using the end-to-end workflow, from data preparation to model serving.

This demo uses the [OSS experimentation platform](https://gitlab.fokus.fraunhofer.de/iml4e/iml4e_oss_exp_platform/-/tree/main).
Clone the OSS experimentation platform [repository](https://gitlab.fokus.fraunhofer.de/iml4e/iml4e_oss_exp_platform/-/tree/main)
and follow the instructions to install the platform locally. (*When installing the OSS experimentation platform, remember to install the optional local image registry as well, as it is required for this demo*)

1. [Model training pipeline overview](./1-Model-training-pipeline-overview.md): How to build a model training pipeline with [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/introduction/).
2. [Build and train your own model](./2-Build-and-train-your-own-model.md): How to create and train your own custom model.
3. [Model evaluation](./3-Model-evaluation.md): How the evaluation works in the training pipeline.
4. [Deploy the model](./4-Manual-model-deployment.md): How to deploy the model manually to KServe on Kubernetes for serverless inference.