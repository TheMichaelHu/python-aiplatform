# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Optional, Dict

from google.auth import credentials as auth_credentials

from google.cloud import aiplatform
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import pipeline_jobs
from google.cloud.aiplatform._pipeline_based_service import pipeline_based_service

from google.cloud.aiplatform.compat.types import (
    pipeline_state_v1 as gca_pipeline_state_v1,
)

_LOGGER = base.Logger(__name__)

MODEL_COMPARISON_PIPELINE = "model_comparison"
BQML_ARIMA_TRAIN_PIPELINE = "bqml_arima_train"
PROPHET_TRAINER_PIPELINE = "prophet_trainer"
AUTOML_TABULAR_PIPELINE = "automl_tabular"
TABNET_TRAINER_PIPELINE = "tabnet_trainer"
TABNET_HYPERPARAMETER_TUNING_JOB_PIPELINE = "tabnet_hyperparameter_tuning_job"
WIDE_AND_DEEP_TRAINER_PIPELINE = "wide_and_deep_trainer"
WIDE_AND_DEEP_HYPERPARAMETER_TUNING_JOB_PIPELINE = "wide_and_deep_hyparameter_tuning_job"
XGBOOST_TRAINER_PIPELINE = "xgboost_trainer"
XGBOOST_HYPERPARAMETER_TUNING_JOB_PIPELINE = "xgboost_hyperparameter_tuning_job"
LEARN_TO_LEARN_FORECASTING_PIPELINE = "learn_to_learn_forecasting"
TIME_SERIES_DENSE_ENCODER_FORECASTING_PIPELINE = "time_series_dense_encoder_forecasting"

_GITHUB_TEMPLATE = (
    "https://raw.githubusercontent.com/kubeflow/pipelines/"
    "google-cloud-pipeline-components-{{version}}/components/google-cloud/"
    "google_cloud_pipeline_components/{stage}/automl/{template}")
_TEMPLATE_GALLERY_TEMPLATE = (
    "https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/{template}"
    "/{{version}}")
_PIPELINE_TEMPLATES = {
    # AUTOML_TABULAR_PIPELINE: (
        # _TEMPLATE_GALLERY_TEMPLATE.format(template="automl-tabular")),
    # TABNET_TRAINER_PIPELINE: (
        # _TEMPLATE_GALLERY_TEMPLATE.format(
            # template="automl-tabular-tabnet-trainer")),
    # TABNET_HYPERPARAMETER_TUNING_JOB_PIPELINE: (
        # _TEMPLATE_GALLERY_TEMPLATE.format(
            # template="automl-tabular-tabnet-hyperparameter-tuning-job")),
    # WIDE_AND_DEEP_TRAINER_PIPELINE: (
        # _TEMPLATE_GALLERY_TEMPLATE.format(
            # template="automl-tabular-wide-and-deep-trainer")),
    # WIDE_AND_DEEP_HYPERPARAMETER_TUNING_JOB_PIPELINE: (
        # _TEMPLATE_GALLERY_TEMPLATE.format(
            # template="automl-tabular-wide-and-deep-hyperparameter-tuning-job")),
    AUTOML_TABULAR_PIPELINE: (
        _GITHUB_TEMPLATE.format(
            stage="v1",
            template="tabular/automl_tabular_pipeline.yaml")),
    TABNET_TRAINER_PIPELINE: (
        _GITHUB_TEMPLATE.format(
            stage="preview",
            template="tabular/tabnet_trainer_pipeline.yaml")),
    TABNET_HYPERPARAMETER_TUNING_JOB_PIPELINE: (
        _GITHUB_TEMPLATE.format(
            stage="preview",
            template="tabular/tabnet_hyperparameter_tuning_job_pipeline.yaml")),
    WIDE_AND_DEEP_TRAINER_PIPELINE: (
        _GITHUB_TEMPLATE.format(
            stage="preview",
            template="tabular/wide_and_deep_trainer_pipeline.yaml")),
    WIDE_AND_DEEP_HYPERPARAMETER_TUNING_JOB_PIPELINE: (
        _GITHUB_TEMPLATE.format(
            stage="preview",
            template="tabular"
            "/wide_and_deep_hyperparameter_tuning_job_pipeline.yaml")),
    XGBOOST_HYPERPARAMETER_TUNING_JOB_PIPELINE: (
        _GITHUB_TEMPLATE.format(
            stage="preview",
            template="tabular"
            "/xgboost_hyperparameter_tuning_job_pipeline.yaml")),
    XGBOOST_TRAINER_PIPELINE: (
        _GITHUB_TEMPLATE.format(
            stage="preview",
            template="tabular/xgboost_trainer_pipeline.yaml")),
    # Experimental pipelines aren't in GCPC.
    MODEL_COMPARISON_PIPELINE: (
        "gs://automl-tables-model-comparison/pipelines/tabular"
        "/model_comparison_pipeline.json"),
    BQML_ARIMA_TRAIN_PIPELINE: (
        _GITHUB_TEMPLATE.format(
            stage="v1",
            template="forecasting/bqml_arima_train_pipeline.yaml")),
    PROPHET_TRAINER_PIPELINE: (
        _GITHUB_TEMPLATE.format(
            stage="v1",
            template="forecasting/prophet_train_pipeline.yaml")),
    LEARN_TO_LEARN_FORECASTING_PIPELINE: (
        _GITHUB_TEMPLATE.format(
            stage="preview",
            template="forecasting"
            "/learn_to_learn_forecasting_pipeline.yaml")),
    TIME_SERIES_DENSE_ENCODER_FORECASTING_PIPELINE: (
        _GITHUB_TEMPLATE.format(
            stage="preview",
            template="forecasting"
            "/time_series_dense_encoder_forecasting_pipeline.yaml")),
}


class ModelComparisonJob(pipeline_based_service._VertexAiPipelineBasedService):

    _template_ref = _PIPELINE_TEMPLATES

    _creation_log_message = "Created PipelineJob for your Model Comparison."

    _component_identifier = "fpc-structured-data"

    _template_name_identifier = "model-comparison"

    @property
    def _metadata_output_artifact(self) -> Optional[str]:
        """The resource uri for the ML Metadata output artifact from the get-experiment of the Model Comparison pipeline"""
        if self.state == gca_pipeline_state_v1.PipelineState.PIPELINE_STATE_SUCCEEDED:
            for task in self.backing_pipeline_job._gca_resource.job_detail.task_details:
                if (
                    task.task_name.startswith("get-experiment")
                    and "experiment" in task.outputs
                ):
                    return (
                        task.outputs["experiment"]
                        .artifacts[0]
                        .metadata["experiment_name"]
                    )

    def __init__(
        self,
        comparison_pipeline_run_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Retrieves a ModelComparisonJob and instantiates its representation.
        Example Usage:
            my_comparison = aiplatform.ModelComparisonJob(
                comparison_pipeline_run_name = "projects/123/locations/us-central1/pipelineJobs/456"
            )
        Args:
            comparison_pipeline_run_name (str):
                Required. A fully-qualified pipeline job run ID.
                Example: "projects/123/locations/us-central1/pipelineJobs/456" or
                "456" when project and location are initialized or passed.
            project (str):
                Optional. Project to retrieve pipeline job from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve pipeline job from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve this pipeline job. Overrides
                credentials set in aiplatform.init.
        """
        super().__init__(
            pipeline_job_name=comparison_pipeline_run_name,
            project=project,
            location=location,
            credentials=credentials,
        )

    @staticmethod
    def get_template_url(
        pipeline: str,
        version: str = "2.3.0",
    ) -> str:
        """Gets the pipeline template URL for a given pipeline.

        Args:
            pipeline (str):
                Required. Pipeline name.
            version (str):
                Optional. Google Cloud Pipeline Components version that hosts
                the pipeline.

        Returns:
            (str): The pipeline template URL.
        """

        template_uri = ModelComparisonJob._template_ref.get(pipeline)
        return template_uri.format(version=version)

    @classmethod
    def submit(
        cls,
        prediction_type: str,
        training_jobs: Dict[str, Dict[str, Any]],
        pipeline_root: str,
        data_source_csv_filenames: str = "",
        data_source_bigquery_table_path: str = "",
        evaluation_data_source_csv_filenames: str = "",
        evaluation_data_source_bigquery_table_path: str = "",
        job_id: Optional[str] = None,
        comparison_pipeline_display_name: Optional[str] = None,
        service_account: Optional[str] = "",
        network: Optional[str] = "",
        project: Optional[str] = None,
        location: Optional[str] = None,
        experiment: Optional[str] = "",
    ) -> "ModelComparisonJob":
        """Submits a Model Comparison Job using aiplatform.PipelineJob and returns
        the ModelComparisonJob resource.

        Example usage:
        my_comparison = _ModelComparisonJob.submit(
            prediction_type="forecasting",
            training_jobs={},
            data_source_bigquery_table_path="bq://test-data.train",
            evaluation_data_source_bigquery_table_path="bq://test-data.eval",
            pipeline_root="gs://my-bucket/pipeline-root/"
        )

        Args:
            prediction_type (str):
                Required. The type of problem being solved. Can be one of:
                regression, classification, or forecasting.
            training_jobs:
                Required. A dict mapping name to a dict of training job inputs.
            pipeline_root (str):
                Required. The GCS directory to store output from the model comparison PipelineJob.
            data_source_csv_filenames (str):
                Optional. Comma-separated paths to CSVs stored in GCS to
                use as the training dataset for all training pipelines. This should be
                None if `data_source_bigquery_table_path` is not None. This should only
                contain data from the training and validation split and not from the test
                split.
            data_source_bigquery_table_path (str):
                Optional. Path to BigQuery Table to use as the
                training dataset for all training pipelines. This should be None if
                `data_source_csv_filenames` is not None. This should only contain data
                from the training and validation split and not from the test split.
            evaluation_data_source_csv_filenames (str):
                Optional. Comma-separated paths to CSVs stored
                in GCS to use as the evaluation dataset for all training pipelines. This
                should be None if `evaluation_data_source_bigquery_table_path` is not
                None. This should only contain data from the test split and not from the
                training and validation split.
            evaluation_data_source_bigquery_table_path (str):
                Optional. Path to BigQuery Table to use as
                the evaluation dataset for all training pipelines. This should be None if
                `evaluation_data_source_csv_filenames` is not None. This should only
                contain data from the test split and not from the training and validation
                split.
            job_id (str):
                Optional. The unique ID of the job run.
                If not specified, pipeline name + timestamp will be used.
            comparison_pipeline_display_name (str):
                Optional. The user-defined name of the PipelineJob created by this Pipeline Based Service.
            service_account (str):
                Optional. Specifies the service account for workload run-as account for this Model
                Comparison PipelineJob and its subpipelines.
                Users submitting jobs must have act-as permission on this run-as account.
                The service account running
                this Model Comparison job needs the following permissions: Storage Admin, Vertex AI User.
            network (str):
                Optional. The full name of the Compute Engine network to which the job and sub pipelines
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
                If left unspecified, the job is not peered with any network.
            project (str):
                Optional. The project to run this PipelineJob in. If not set,
                the project set in aiplatform.init will be used.
            location (str):
                Optional. Location to create PipelineJob. If not set,
                location set in aiplatform.init will be used.
            experiment (str):
                Optional. The Vertex AI experiment name to be used for this model comparison job.
                If not provided will be auto-generated.
        Returns:
            (ModelComparisonJob): Instantiated representation of the model comparison job.
        """

        if bool(data_source_csv_filenames) == bool(data_source_bigquery_table_path):
            raise ValueError(
                "Exactly one data source parameter should be set. data_source_csv_filenames=%s, data_source_bigquery_table_path=%s"
                % (data_source_csv_filenames, data_source_bigquery_table_path)
            )

        if bool(evaluation_data_source_csv_filenames) == bool(
            evaluation_data_source_bigquery_table_path
        ):
            raise ValueError(
                "Exactly one data source parameter should be set. evaluation_data_source_csv_filenames=%s, evaluation_data_source_bigquery_table_path=%s"
                % (
                    evaluation_data_source_csv_filenames,
                    evaluation_data_source_bigquery_table_path,
                )
            )

        if not comparison_pipeline_display_name:
            comparison_pipeline_display_name = cls._generate_display_name()

        template_params = {
            "project": project or initializer.global_config.project,
            "location": location or initializer.global_config.location,
            "root_dir": pipeline_root,
            "prediction_type": prediction_type,
            "training_jobs": training_jobs,
            "data_source_csv_filenames": data_source_csv_filenames,
            "data_source_bigquery_table_path": data_source_bigquery_table_path,
            "evaluation_data_source_csv_filenames": evaluation_data_source_csv_filenames,
            "evaluation_data_source_bigquery_table_path": evaluation_data_source_bigquery_table_path,
            "experiment": experiment,
            "service_account": service_account,
            "network": network,
        }

        template_url = cls.get_template_url(MODEL_COMPARISON_PIPELINE)

        comparison_pipeline_run = cls._create_and_submit_pipeline_job(
            template_params=template_params,
            template_path=template_url,
            pipeline_root=pipeline_root,
            display_name=comparison_pipeline_display_name,
            job_id=job_id,
            service_account=service_account,
            network=network,
            project=project,
            location=location,
        )

        _LOGGER.info(
            f"{ModelComparisonJob._creation_log_message} View it in the console: {comparison_pipeline_run.pipeline_console_uri}"
        )

        return comparison_pipeline_run

    def get_model_comparison_results(
        self,
    ) -> Optional["pd.DataFrame"]:  # noqa: F821
        """Gets the DataFrame with the results of the model comparison.
        Returns:
            pd.DataFrame: A dataframe with the results of the model comparison job.
        Raises:
            RuntimeError: If the ModelComparisonJobb pipeline failed.
        """
        eval_job_state = self.backing_pipeline_job.state

        if eval_job_state in pipeline_jobs._PIPELINE_ERROR_STATES:
            raise RuntimeError(
                f"Comparison job failed. For more details see the logs: {self.pipeline_console_uri}"
            )
        elif eval_job_state not in pipeline_jobs._PIPELINE_COMPLETE_STATES:
            _LOGGER.info(
                f"Your comparison job is still in progress. For more details see the logs {self.pipeline_console_uri}"
            )
        else:
            return aiplatform.Experiment(
                experiment_name=self._metadata_output_artifact
            ).get_data_frame()
