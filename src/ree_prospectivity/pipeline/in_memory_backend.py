"""Reference backend that wires every computational stage in memory."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ree_prospectivity.augmentation import AugmentationResult, generate_augmented_samples
from ree_prospectivity.data.contracts import SampleBatch
from ree_prospectivity.evaluation import ClassificationReport, evaluate_classifier
from ree_prospectivity.inference import predict_prospectivity_grid
from ree_prospectivity.models import (
    PatchDiscriminator,
    ProspectivityCNN,
    SRGenerator,
)
from ree_prospectivity.pipeline.artifacts import (
    ArtifactKind,
    ArtifactRef,
    StageOutput,
)
from ree_prospectivity.pipeline.plan import PipelineStage
from ree_prospectivity.pipeline.runner import StageContext
from ree_prospectivity.preprocessing import (
    ChannelMinMaxScaler,
    FeatureCube,
    LabeledSite,
    extract_site_windows,
)
from ree_prospectivity.splitting import (
    DatasetSplits,
    paper_random_split,
    stratified_group_split,
)
from ree_prospectivity.training import (
    ClassifierTrainingResult,
    GANTrainingResult,
    train_classifier,
    train_gan,
)


def _scale_batch(batch: SampleBatch, scaler: ChannelMinMaxScaler) -> SampleBatch:
    return replace(batch, high_resolution=scaler.transform(batch.high_resolution))


def _artifact(
    kind: ArtifactKind,
    *,
    shape: tuple[int, ...] | None = None,
    sample_count: int | None = None,
    confidential: bool = True,
) -> ArtifactRef:
    metadata: dict[str, object] = {"schema_version": "1"}
    if shape is not None:
        metadata["shape"] = shape
    if sample_count is not None:
        metadata["sample_count"] = sample_count
    return ArtifactRef(
        kind=kind,
        uri=f"memory://{kind.value}",
        confidential=confidential,
        metadata=metadata,
    )


class InMemoryResearchBackend:
    """Complete reference flow; file-system adapters can replace only ingest/export."""

    def __init__(
        self,
        feature_cube: FeatureCube,
        sites: tuple[LabeledSite, ...],
        *,
        device: str = "cpu",
    ) -> None:
        self.feature_cube = feature_cube
        self.sites = sites
        self.device = device
        self.windows: SampleBatch | None = None
        self.scaler: ChannelMinMaxScaler | None = None
        self.splits: DatasetSplits | None = None
        self.gan_result: GANTrainingResult | None = None
        self.augmentation_result: AugmentationResult | None = None
        self.classifier_result: ClassifierTrainingResult | None = None
        self.classification_report: ClassificationReport | None = None
        self.prospectivity_grid: np.ndarray | None = None

    def execute(self, stage: PipelineStage, context: StageContext) -> StageOutput:
        handlers = {
            PipelineStage.INGEST: self._ingest,
            PipelineStage.PREPARE: self._prepare,
            PipelineStage.SPLIT: self._split,
            PipelineStage.TRAIN_GAN: self._train_gan,
            PipelineStage.AUGMENT: self._augment,
            PipelineStage.VALIDATE_AUGMENTATION: self._validate_augmentation,
            PipelineStage.TRAIN_CLASSIFIER: self._train_classifier,
            PipelineStage.EVALUATE: self._evaluate,
            PipelineStage.PREDICT: self._predict,
            PipelineStage.EXPORT: self._export,
        }
        return handlers[stage](context)

    def _ingest(self, context: StageContext) -> StageOutput:
        del context
        return StageOutput(
            artifacts=(
                _artifact(
                    ArtifactKind.RAW_FEATURE_CUBE,
                    shape=tuple(self.feature_cube.values.shape),
                ),
                _artifact(ArtifactKind.SITE_LABELS, sample_count=len(self.sites)),
            ),
            metrics={
                "height": self.feature_cube.values.shape[0],
                "width": self.feature_cube.values.shape[1],
                "channels": self.feature_cube.values.shape[2],
                "site_count": len(self.sites),
            },
        )

    def _prepare(self, context: StageContext) -> StageOutput:
        config = context.config
        if self.feature_cube.values.shape[-1] != config.data.channels:
            raise ValueError("feature cube does not match configured channel count")
        self.windows = extract_site_windows(
            self.feature_cube,
            self.sites,
            window_size=config.preprocessing.window_size,
            border_policy=config.preprocessing.border_policy,
        )
        return StageOutput(
            artifacts=(
                _artifact(
                    ArtifactKind.PREPARED_WINDOWS,
                    shape=tuple(self.windows.high_resolution.shape),
                    sample_count=len(self.windows.labels),
                ),
            ),
            metrics={"prepared_site_count": len(self.windows.labels)},
        )

    def _split(self, context: StageContext) -> StageOutput:
        config = context.config
        if config.split_policy == "paper_faithful":
            if self.augmentation_result is None:
                raise RuntimeError("paper-faithful split requires augmented samples")
            self.splits = paper_random_split(
                self.augmentation_result.samples,
                validation_fraction=config.split.validation_fraction,
                seed=config.seed,
            )
            warning = (
                "Post-augmentation random split can leak parent-site information.",
            )
        else:
            if self.windows is None:
                raise RuntimeError("prepared windows are unavailable")
            self.splits = stratified_group_split(
                self.windows,
                validation_fraction=config.split.validation_fraction,
                test_fraction=config.split.test_fraction,
                seed=config.seed,
            )
            warning = ()
        return StageOutput(
            artifacts=(_artifact(ArtifactKind.SPLIT_MANIFEST),),
            metrics={
                "train_samples": len(self.splits.train.labels),
                "validation_samples": len(self.splits.validation.labels),
                "test_samples": (
                    len(self.splits.test.labels) if self.splits.test is not None else 0
                ),
            },
            warnings=warning,
        )

    def _train_gan(self, context: StageContext) -> StageOutput:
        config = context.config
        if self.windows is None:
            raise RuntimeError("prepared windows are unavailable")
        if config.split_policy == "group_before_augmentation":
            if self.splits is None:
                raise RuntimeError("leakage-safe GAN training requires an existing split")
            gan_source = self.splits.train
        else:
            gan_source = self.windows

        self.scaler = ChannelMinMaxScaler.fit(gan_source.high_resolution)
        scaled_source = _scale_batch(gan_source, self.scaler)
        if self.splits is not None and config.split_policy == "group_before_augmentation":
            self.splits = DatasetSplits(
                train=scaled_source,
                validation=_scale_batch(self.splits.validation, self.scaler),
                test=(
                    _scale_batch(self.splits.test, self.scaler)
                    if self.splits.test is not None
                    else None
                ),
            )

        generator = SRGenerator(
            channels=config.data.channels,
            features=config.generator.features,
            residual_blocks=config.generator.residual_blocks,
            scale_factor=config.generator.scale_factor,
        )
        discriminator = PatchDiscriminator(
            channels=config.data.channels,
            base_features=config.discriminator.base_features,
        )
        self.gan_result = train_gan(
            scaled_source.high_resolution,
            generator,
            discriminator,
            downscale_config=config.downscale,
            training=config.gan_training,
            seed=config.seed,
            adversarial_weight=config.gan_loss.adversarial_weight,
            reconstruction_weight=config.gan_loss.reconstruction_weight,
            device=self.device,
        )
        final_metrics = self.gan_result.history[-1]
        return StageOutput(
            artifacts=(
                _artifact(ArtifactKind.PREPROCESSING_STATE),
                _artifact(ArtifactKind.GENERATOR_CHECKPOINT),
                _artifact(ArtifactKind.DISCRIMINATOR_CHECKPOINT),
            ),
            metrics={
                "epochs": len(self.gan_result.history),
                "generator_loss": final_metrics.generator_loss,
                "discriminator_loss": final_metrics.discriminator_loss,
            },
        )

    def _augment(self, context: StageContext) -> StageOutput:
        config = context.config
        if self.gan_result is None or self.scaler is None or self.windows is None:
            raise RuntimeError("GAN, scaler, or prepared windows are unavailable")
        if config.split_policy == "group_before_augmentation":
            if self.splits is None:
                raise RuntimeError("leakage-safe augmentation requires splits")
            source = self.splits.train
        else:
            source = _scale_batch(self.windows, self.scaler)
        self.augmentation_result = generate_augmented_samples(
            source,
            self.gan_result.generator,
            samples_per_site=config.samples_per_site,
            downscale_config=config.downscale,
            seed=config.seed,
            data_range=config.evaluation.psnr_data_range,
            device=self.device,
        )
        return StageOutput(
            artifacts=(
                _artifact(
                    ArtifactKind.AUGMENTED_WINDOWS,
                    shape=tuple(self.augmentation_result.samples.high_resolution.shape),
                    sample_count=len(self.augmentation_result.samples.labels),
                ),
            ),
            metrics={
                "augmented_samples": len(self.augmentation_result.samples.labels),
                "mean_psnr": self.augmentation_result.mean_psnr,
            },
        )

    def _validate_augmentation(self, context: StageContext) -> StageOutput:
        del context
        if self.augmentation_result is None:
            raise RuntimeError("augmentation results are unavailable")
        psnr = self.augmentation_result.psnr_by_sample
        return StageOutput(
            artifacts=(
                _artifact(
                    ArtifactKind.AUGMENTATION_REPORT,
                    confidential=False,
                ),
            ),
            metrics={
                "psnr_mean": float(psnr.mean()),
                "psnr_median": float(np.median(psnr)),
                "psnr_min": float(psnr.min()),
                "psnr_max": float(psnr.max()),
            },
        )

    def _train_classifier(self, context: StageContext) -> StageOutput:
        config = context.config
        if self.splits is None or self.augmentation_result is None:
            raise RuntimeError("splits and augmentation results are required")
        if config.split_policy == "paper_faithful":
            train_samples = self.splits.train
        else:
            train_samples = self.augmentation_result.samples
        model = ProspectivityCNN(
            channels=config.data.channels,
            input_size=config.data.high_resolution_size,
            filters=config.classifier.filters,
            hidden_dims=config.classifier.hidden_dims,
            dropout=config.classifier.dropout,
        )
        self.classifier_result = train_classifier(
            model,
            train_samples,
            self.splits.validation,
            training=config.classifier_training,
            seed=config.seed,
            device=self.device,
        )
        best = self.classifier_result.history[self.classifier_result.best_epoch - 1]
        return StageOutput(
            artifacts=(_artifact(ArtifactKind.CLASSIFIER_CHECKPOINT),),
            metrics={
                "best_epoch": self.classifier_result.best_epoch,
                "validation_loss": best.validation_loss,
                "validation_accuracy": best.validation_accuracy,
            },
        )

    def _evaluate(self, context: StageContext) -> StageOutput:
        if self.classifier_result is None or self.splits is None:
            raise RuntimeError("classifier and splits are unavailable")
        evaluation_samples = self.splits.test or self.splits.validation
        self.classification_report = evaluate_classifier(
            self.classifier_result.model,
            evaluation_samples,
            threshold=context.config.evaluation.classification_threshold,
            device=self.device,
        )
        return StageOutput(
            artifacts=(
                _artifact(ArtifactKind.EVALUATION_REPORT, confidential=False),
            ),
            metrics=self.classification_report.as_public_metrics(),
            warnings=(
                ("No independent test split; metrics use validation samples.",)
                if self.splits.test is None
                else ()
            ),
        )

    def _predict(self, context: StageContext) -> StageOutput:
        if self.classifier_result is None or self.scaler is None:
            raise RuntimeError("classifier and preprocessing state are unavailable")
        self.prospectivity_grid = predict_prospectivity_grid(
            self.feature_cube.values,
            self.classifier_result.model,
            window_size=context.config.inference.window_size,
            batch_size=context.config.inference.batch_size,
            transform=self.scaler.transform,
            device=self.device,
        )
        return StageOutput(
            artifacts=(
                _artifact(
                    ArtifactKind.PROSPECTIVITY_GRID,
                    shape=tuple(self.prospectivity_grid.shape),
                ),
            ),
            metrics={
                "minimum_probability": float(self.prospectivity_grid.min()),
                "maximum_probability": float(self.prospectivity_grid.max()),
            },
        )

    def _export(self, context: StageContext) -> StageOutput:
        if self.prospectivity_grid is None or self.classification_report is None:
            raise RuntimeError("prediction grid and evaluation report are unavailable")
        threshold = context.config.evaluation.classification_threshold
        prospective_fraction = float((self.prospectivity_grid >= threshold).mean())
        return StageOutput(
            artifacts=(
                _artifact(ArtifactKind.PROSPECTIVITY_GRID),
                _artifact(ArtifactKind.PUBLIC_SUMMARY, confidential=False),
            ),
            metrics={
                **self.classification_report.as_public_metrics(),
                "prospective_area_fraction": prospective_fraction,
            },
            warnings=(
                "In-memory export returns references; the local backend must write "
                "reviewed geospatial files.",
            ),
        )
