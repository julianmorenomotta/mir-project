from types import SimpleNamespace
from typing import Any, Dict, Optional, TypedDict

import torch
from torch import nn, optim
import torchmetrics as tm


class OperationMode:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"


RawInputType = Dict


def nested_dict_to_nested_namespace(d: dict) -> SimpleNamespace:
    d_ = d.copy()

    for k, v in d.items():
        if isinstance(v, dict):
            v = nested_dict_to_nested_namespace(v)

        d_[k] = v

    return SimpleNamespace(**d_)


RawInputType = TypedDict(
    "RawInputType",
    {
        "mixture": torch.Tensor,
        "sources": Dict[str, torch.Tensor],
        "estimates": Optional[Dict[str, torch.Tensor]],
        "metadata": Dict[str, Any],
    },
    total=False,
)


def input_dict(
    mixture: torch.Tensor = None,
    sources: Dict[str, torch.Tensor] = None,
    query: torch.Tensor = None,
    metadata: Dict[str, Any] = None,
    modality: str = "audio",
) -> RawInputType:

    def _to_owned_float_tensor(x: Any) -> torch.Tensor:
        # Clone to detach from NumPy-backed, non-resizable storage so default
        # DataLoader collation can allocate shared-memory batches safely.
        return torch.as_tensor(x, dtype=torch.float32).clone().contiguous()

    out = {
        # Estimates are produced by the model later in the pipeline.
        # Keep placeholders tensor-free so DataLoader collation does not
        # attempt to stack zero-length tensors across worker processes.
        "estimates": {k: {} for k in (sources or {}).keys()}
    }

    if mixture is not None:
        out["mixture"] = {modality: _to_owned_float_tensor(mixture)}

    if sources is not None:
        out["sources"] = {
            k: {modality: _to_owned_float_tensor(v)} for k, v in sources.items()
        }

    if query is not None:
        out["query"] = {modality: _to_owned_float_tensor(query)}

    if metadata is not None:
        out["metadata"] = metadata

    return out


class SimpleishNamespace(SimpleNamespace):
    def __init__(self, **kwargs: Any) -> None:
        kwargs_ = kwargs.copy()

        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = SimpleishNamespace(**v)

            kwargs_[k] = v

        super().__init__(**kwargs_)

    def copy(self) -> "SimpleishNamespace":
        return SimpleishNamespace(**{k: v for k, v in self.__dict__.items()})

    def add_subnamespace(self, name: str, **kwargs: Any) -> None:
        if hasattr(self, name):
            raise ValueError(f"Namespace already has attribute {name}")

        setattr(self, name, SimpleishNamespace(**kwargs))

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def items(self):
        return self.__dict__.items()


class BatchedInputOutput(SimpleishNamespace):
    mixture: torch.Tensor
    sources: Dict[str, torch.Tensor]
    estimates: Optional[Dict[str, torch.Tensor]]
    metadata: Dict[str, Any]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "BatchedInputOutput":
        return cls(**d)

    def to_dict(self) -> dict:
        return self.__dict__


class TensorCollection(SimpleishNamespace):
    def __init__(self, **kwargs: torch.Tensor) -> None:
        super().__init__(**kwargs)

    def apply(self, func: Any, *args: Any, **kwargs: Any) -> "TensorCollection":
        return TensorCollection(
            **{k: func(v, *args, **kwargs) for k, v in self.__dict__.items()}
        )

    def as_stacked_tensor(self, dim: int = 0) -> torch.Tensor:
        return torch.stack(list(self.__dict__.values()), dim=dim)

    def as_concatenated_tensor(self, dim: int = 0) -> torch.Tensor:
        return torch.cat(list(self.__dict__.values()), dim=dim)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.__dict__[key]


InputType = BatchedInputOutput
OutputType = BatchedInputOutput
LossOutputType = Any
MetricOutputType = Any

ModelType = nn.Module
OptimizerType = optim.Optimizer
SchedulerType = optim.lr_scheduler._LRScheduler
MetricType = tm.Metric
LossType = nn.Module

OptimizationBundle = Any

LossHandler = Any
MetricHandler = Any
AugmentationHandler = Any
InferenceHandler = Any
