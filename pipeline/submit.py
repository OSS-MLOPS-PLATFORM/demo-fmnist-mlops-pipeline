import json
import logging
import typing
from dataclasses import dataclass

import click
import kfp
import kfp_server_api

from .config import DEFAULT_RUN_PIPELINE_FILE, get_config
from .auth_kubeflow import get_istio_auth_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArgoWorkflowDetails:
    name: str
    namespace: str


def _parse_argo_workflow(
    run: kfp_server_api.ApiRunDetail,
) -> typing.Optional[ArgoWorkflowDetails]:
    """
    Parse Argo workflow name. Allow this to fail because other
    workflow engines may be supported in the future.
    """
    try:
        pipeline_runtime = run.pipeline_runtime
        workflow_manifest = json.loads(pipeline_runtime.workflow_manifest)
        metadata = workflow_manifest["metadata"]
        name = metadata["name"]
        namespace = metadata["namespace"]
        return ArgoWorkflowDetails(name=name, namespace=namespace)
    except AttributeError:
        pass


def _display_helper_commands(run: kfp_server_api.ApiRunDetail):
    argo_workflow = _parse_argo_workflow(run=run)
    run_id = run.run.id

    logger.info(f"Execute 'kfp run get {run_id} --watch' to watch the job with KFP CLI")

    if argo_workflow:
        logger.info(
            f"Execute 'argo watch {argo_workflow.name} -n {argo_workflow.namespace}' to"
            f" watch the job"
        )
        logger.info(
            f"Execute 'argo logs {argo_workflow.name} -n {argo_workflow.namespace} "
            f"--follow' to follow logs"
        )


def _handle_job_end(run_detail: kfp_server_api.ApiRunDetail):
    finished_run = run_detail.to_dict()["run"]

    created_at = finished_run["created_at"]
    finished_at = finished_run["finished_at"]

    duration_secs = (finished_at - created_at).total_seconds()

    status = finished_run["status"]

    logger.info(f"Run finished in {round(duration_secs)} seconds with status: {status}")

    if status != "Succeeded":
        raise Exception(f"Run failed: {run_detail.run.id}")


def run_pipeline(
    kfp_host: str,
    pipeline_file: str,
    mlflow_tracking_uri: str,
    mlflow_s3_endpoint_url: str,
    experiment: str,
    model_registry_name: str,
    kubeflow_url: str,
    kubeflow_username: str,
    kubeflow_password: str,
    namespace: str,
    wait: bool,
    register_model: bool,
    run_name: typing.Optional[str] = None,
):
    logger.info(
        f"Running pipeline from file: {pipeline_file} with run name: {run_name}"
    )
    auth_session = get_istio_auth_session(
        url=kubeflow_url,
        username=kubeflow_username,
        password=kubeflow_password
    )
    logger.info(f"Connecting to Kubeflow Pipelines at {kfp_host}")
    client = kfp.Client(
        host=f"{kubeflow_url}/pipeline",
        cookies=auth_session["session_cookie"]
    )
    created_run = client.create_run_from_pipeline_package(
        pipeline_file=pipeline_file,
        arguments={
            "experiment": experiment,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_s3_endpoint_url": mlflow_s3_endpoint_url,
            "model_registry_name": model_registry_name,
            "register_model": str(register_model),
        },
        enable_caching=False,
        run_name=run_name,
        experiment_name=experiment,
        namespace=namespace,
    )

    run_id = created_run.run_id

    logger.info(f"Submitted run with ID: {run_id}")

    run = client.get_run(run_id=run_id)

    # client.get_run should return `ApiRun` but it seems to return `ApiRunDetail` 🤷‍♂️
    # so ignore the type for now
    _display_helper_commands(run=run)  # type: ignore
    if wait:
        logger.info(f"Waiting for run {run_id} to complete....")
        run_detail = created_run.wait_for_run_completion()
        _handle_job_end(run_detail)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.option(
    "--pipeline-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default=DEFAULT_RUN_PIPELINE_FILE,
)
@click.option(
    "--register/--no-register",
    is_flag=True,
    default=False,
    type=bool,
    help="Whether to run model registration step",
)
@click.option(
    "-w",
    "--wait/--no-wait",
    is_flag=True,
    default=True,
    type=bool,
    help="""Whether to wait until the Kubeflow pipeline finishes execution""",
)
@click.argument(
    "overrides",
    nargs=-1,
    type=click.UNPROCESSED
)
@click.option(
    "--namespace",
    type=str,
    default="kubeflow-user-example-com",
    help="Name of the user namespace to run the pipeline in",
)
@click.option(
    "--kubeflow-url",
    type=str,
    default="http://localhost:8080",
    help="URL to Kubeflow",
)
@click.option(
    "--kubeflow-username",
    type=str,
    default="user@example.com",
    help="Kubeflow username",
)
@click.option(
    "--kubeflow-password",
    type=str,
    default="12341234",
    help="Kubeflow password",
)
def cli(
    pipeline_file: str,
    register: bool,
    namespace: str,
    kubeflow_url: str,
    kubeflow_username: str,
    kubeflow_password: str,
    wait: bool,
    overrides: list = None,
):
    """Submit and run a Kubeflow Pipeline file"""
    logging.basicConfig(level=logging.INFO)

    cfg = get_config(overrides=overrides)

    run_pipeline(
        kfp_host=cfg.submit_config.host,
        pipeline_file=pipeline_file,
        mlflow_tracking_uri=cfg.submit_config.mlflow_tracking_uri,
        mlflow_s3_endpoint_url=cfg.submit_config.mlflow_s3_endpoint_url,
        experiment=cfg.submit_config.experiment,
        model_registry_name=cfg.submit_config.model_registry_name,
        run_name=cfg.submit_config.run,
        namespace=namespace,
        kubeflow_url=kubeflow_url,
        kubeflow_username=kubeflow_username,
        kubeflow_password=kubeflow_password,
        wait=wait,
        register_model=register,
    )


if __name__ == "__main__":
    cli()
