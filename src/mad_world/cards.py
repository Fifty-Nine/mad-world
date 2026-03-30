"""Basic infrastructure for polymorphic Card classes."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from pydantic import (
    BaseModel,
    Field,
    ValidatorFunctionWrapHandler,
    model_validator,
)


class CardNameCollisionError(Exception):
    def __init__(self, name: str, exist_name: str) -> None:
        super().__init__(
            f"Card with name {name} already exists in registry; "
            f"Existing class name: {exist_name}"
        )


class BaseCard(BaseModel):
    """The common base type for all cards in the game."""

    card_kind: str = Field(
        description="The concrete kind of card represented by this card."
    )

    _registries: ClassVar[dict[type, dict[str, type[BaseCard]]]] = {}

    @classmethod
    def __init_subclass__(
        cls,
        *,
        is_base: bool = False,
        card_kind: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Add a subclass to the Card registry."""
        super().__init_subclass__(**kwargs)

        if is_base:
            cls._registries[cls] = {}
            return

        if card_kind is None:
            return

        for base_class in cls.__mro__:
            registry = cls._registries.get(base_class)
            if registry is None:
                continue

            if card_kind not in registry:
                registry[card_kind] = cls
                break

            raise CardNameCollisionError(
                card_kind, registry[card_kind].__name__
            )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, BaseCard):
            return NotImplemented

        return self.card_kind < other.card_kind

    @model_validator(mode="wrap")
    @classmethod
    def route_to_subclass(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> BaseCard:
        registry = cls._registries.get(cls)
        card_kind = getattr(value, "card_kind", None)
        if card_kind is None and isinstance(value, dict):
            card_kind = value.get("card_kind")

        if not isinstance(card_kind, str):
            return cast("BaseCard", handler(value))

        concrete_type = (
            registry.get(card_kind) if registry is not None else None
        )
        if concrete_type is None:
            return cast("BaseCard", handler(value))

        return concrete_type.model_validate(value)
