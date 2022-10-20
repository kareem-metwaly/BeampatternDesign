"""Contains all implementations of items such as: DatasetItem(s), ModelOutput,
LossItem, TotalLoss."""
import typing as t

import torch
from utils.base_classes import ArbitraryBaseModel
from utils.types import FilePath

# Item Classes ##################


class DatasetItem(ArbitraryBaseModel):
    """implementation of the dataset item containing the optimum waveform and
    the desired beampattern."""

    optimum_waveform: torch.Tensor
    desired_beampattern: torch.Tensor
    filepath: t.Optional[FilePath] = None

    def __post_init__(self, **kwargs):
        """check the types."""
        super().__post_init__(**kwargs)
        assert isinstance(self.optimum_waveform, torch.Tensor)
        assert isinstance(self.desired_beampattern, torch.Tensor)
        # TODO: check kwargs and validate the dimensions based on the parameters if they are given


class DatasetItems(ArbitraryBaseModel):
    """combines multiple DatasetItem and collate them."""

    optimum_waveforms: torch.Tensor
    desired_beampatterns: torch.Tensor
    filepaths: t.Sequence[t.Optional[FilePath]] = []

    def __post_init__(self, **kwargs):
        """check types and dimensions matching."""
        super().__post_init__(**kwargs)
        assert isinstance(self.optimum_waveforms, torch.Tensor)
        assert isinstance(self.desired_beampatterns, torch.Tensor)
        assert self.optimum_waveforms.shape[0] == self.desired_beampatterns.shape[0], (
            f"we must have the same batch size,\ngiven: "
            f"waveforms = {self.optimum_waveforms.shape[0]} and beampattern = {self.desired_beampatterns.shape[0]}"
        )
        # TODO: check kwargs and validate the dimensions based on the parameters if they are given

    @classmethod
    def collate(cls: t.Type["DatasetItems"], items: t.Sequence[DatasetItem], **kwargs) -> "DatasetItems":
        """combining multiple items to have a batch.

        :param items: (Sequence[DatasetItem]) the data to be staged together
        :param kwargs: may contain parameters to check the dimensions
        :return DatasetItems
        """
        return cls(
            optimum_waveforms=torch.stack([item.optimum_waveform for item in items], dim=0),
            desired_beampatterns=torch.stack([item.desired_beampattern for item in items], dim=0),
            filepaths=[item.filepath for item in items],
            **kwargs,
        )

    def to(self, device) -> "DatasetItems":
        self.desired_beampatterns = self.desired_beampatterns.to(device)
        self.optimum_waveforms = self.optimum_waveforms.to(device)
        return self


class ModelOutput(ArbitraryBaseModel):
    estimated_waveforms: torch.Tensor

    def __post_init__(self, **kwargs: t.Any) -> None:
        """Check the dimensions and types."""
        super().__post_init__(**kwargs)


class FinalModelOutput(ModelOutput):
    """The final output of the model must have the estimated waveform."""

    initial: t.Optional[torch.Tensor] = None
    steps: t.Optional[torch.Tensor] = None

    def __post_init__(self, **kwargs: t.Any) -> None:
        """Check the dimensions and types."""
        super().__post_init__(**kwargs)
        assert isinstance(
            self.estimated_waveforms, torch.Tensor
        ), f"Estimated waveform has to be a Tensor, given: {type(self.estimated_waveforms)}"
        assert (
            len(self.estimated_waveforms.shape) == 2
        ), f"The estimated waveforms has to be a 2D Tensor, given shape is {self.estimated_waveforms.shape}"
        if self.initial is not None:
            assert isinstance(self.initial, torch.Tensor), f"initial has to be a Tensor, given: {type(self.initial)}"
            assert len(self.initial.shape) == 3, f"initial has to be a 3D Tensor, given shape is {self.initial.shape}"
        if self.steps is not None:
            assert isinstance(self.steps, torch.Tensor), f"initial has to be a Tensor, given: {type(self.steps)}"
            assert len(self.steps.shape) == 3, f"initial has to be a 3D Tensor, given shape is {self.steps.shape}"


class StepOutput(ModelOutput):
    """The output must have the estimated waveform."""

    estimated_gradients: torch.Tensor
    estimated_steps: torch.Tensor

    def __post_init__(self, **kwargs: t.Any) -> None:
        """Check the dimensions and types."""
        super().__post_init__(**kwargs)
        assert isinstance(
            self.estimated_waveforms, torch.Tensor
        ), f"Estimated waveform has to be a Tensor, given: {type(self.estimated_waveforms)}"
        assert (
            len(self.estimated_waveforms.shape) == 3
        ), f"The estimated waveforms has to be a 3D Tensor, given shape is {self.estimated_waveforms.shape}"
        assert isinstance(
            self.estimated_gradients, torch.Tensor
        ), f"Estimated gradients has to be a Tensor, given: {type(self.estimated_gradients)}"
        assert (
            len(self.estimated_gradients.shape) == 3
        ), f"The estimated gradients has to be a 3D Tensor, given shape is {self.estimated_gradients.shape}"


# Loss related items ####################


class LossItem(ArbitraryBaseModel):
    """defines a term that could be used in backpropagation, logging,
    caching."""

    name: str
    value: torch.Tensor
    isBackpropagated: bool = True
    isCached: bool = False
    isLogged: bool = False
    isImage: bool = False
    weight: t.Optional[t.Union[float, torch.Tensor]] = 1.0

    def __post_init__(self, **kwargs):
        """check consistency."""
        super().__post_init__(**kwargs)
        assert isinstance(self.value, torch.Tensor), f"value should be a Tensor, given: {type(self.value)}"
        self.name = self.name.replace("/", "_").replace("\\", "_")  # as it may be understood as folder
        if self.isBackpropagated:
            assert torch.is_floating_point(self.value), f"{self.name} has type {type(self.value)}"
            assert self.value.numel() == 1, self.value.shape
            assert self.value.requires_grad or not torch.is_grad_enabled()
            assert not self.isImage
            assert not (
                self.value.isnan() or self.value.isinf()
            ), f"loss value is nan or inf; {self.value} for {self.name}"
        if self.isImage:
            self.isLogged = False  # isLogged shouldn't be used with images, images are anyway logged


class TotalLoss:
    """combines multiple LossItems and defines the actual value used for
    backpropagation."""

    _items: t.List[LossItem]
    _names_idx_maps: t.Dict[str, int]

    def __init__(self, items: t.Union[LossItem, t.List[LossItem]]):
        """combines a list of different losses.

        :param items: (LossItem or a list of them) the losses to be stored
        """
        self._items = items
        if not isinstance(self._items, t.List):
            self._items = [items]
        for item in self._items:
            assert isinstance(item, LossItem), f"all items should be of type LossItem, given one is: {type(item)}"
        self._names_idx_maps = {item.name: idx for idx, item in enumerate(self._items)}

    @property
    def names(self) -> t.Iterator[str]:
        """return the names of the LossItems."""
        return iter(self._names_idx_maps.keys())

    @property
    def backpropagated_items(self) -> t.Iterator[LossItem]:
        """Iterator over backpropagated items."""
        return (item for item in self._items if item.isBackpropagated)

    @property
    def cached_items(self) -> t.Iterator[LossItem]:
        """Iterator over cached items."""
        return (item for item in self._items if item.isCached)

    @property
    def logged_items(self) -> t.Iterator[LossItem]:
        """iterator over logged items."""
        return (item for item in self._items if item.isLogged)

    @property
    def logged_images(self) -> t.Iterator[LossItem]:
        """iterator over logged images."""
        return (item for item in self._items if item.isImage)

    def __repr__(self):
        """a good representation of the losses stored."""
        string = [f"{item.name}={item.value}" for item in self._items if item.isBackpropagated]
        return f"{self.__class__.__name__}({string})"

    @property
    def value(self) -> torch.Tensor:
        """Should return the value that will be used later for backward
        propagation."""
        total = 0
        for item in self.backpropagated_items:
            total = total + item.weight * item.value
        return total

    @property
    def logs(self) -> t.Optional[t.Dict[str, float]]:
        """Should return a dict of values to be logged."""
        items = {"logged/" + item.name: item.value for item in self.logged_items}
        items.update({"backpropagated/" + item.name: item.value for item in self.backpropagated_items})
        items.update({k: v.item() for k, v in items.items() if hasattr(v, "item")})
        items.update(
            {k: v.detach() for k, v in items.items() if hasattr(v, "detach")}
        )  # if item is not found for some reason
        return items

    @property
    def cache(self) -> t.Optional[t.Dict[str, t.Union[float, torch.Tensor]]]:
        """should return values to be cached and then averaged over each
        epoch."""
        items = {item.name: item.value for item in self.cached_items}
        # detaching from backprop graph as we are only interested in the values
        for k, v in items.items():
            if hasattr(v, "item") and v.numel() == 1:
                items.update({k: v.item()})
            elif hasattr(v, "detach"):
                items.update({k: v.detach()})
        return items

    def add(self, loss_item: LossItem) -> None:
        """appending a new loss term."""
        assert loss_item.name not in self.names, f"{loss_item.name} and {list(self.names)}"
        self._names_idx_maps[loss_item.name] = len(self._items)
        self._items.append(loss_item)

    def extend(self, losses: t.Union["TotalLoss", t.Sequence[LossItem]]) -> None:
        """extending the current losses with a sequence of losses."""
        for loss_item in losses:
            self.add(loss_item)

    def __iter__(self) -> LossItem:
        """loop through different LossItem."""
        for item in self._items:
            yield item

    def update_by_name(self, name: str, item: LossItem):
        """update the provided name with a new LossItem."""
        idx = self._names_idx_maps[name]
        self._items[idx] = item

    def update_existing(self, new_loss_items: t.Union[LossItem, t.Sequence[LossItem], "TotalLoss"]) -> None:
        """update an already existing item or list of items."""
        if isinstance(new_loss_items, LossItem):
            new_loss_items = [new_loss_items]
        for item in new_loss_items:
            self.update_by_name(item.name, item)
