"""Basic infrastructure for polymorphic Card classes."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, cast

from pydantic import (
    BaseModel,
    ValidatorFunctionWrapHandler,
    model_serializer,
    model_validator,
)


class CardNameCollisionError(Exception):
    def __init__(self, name: str, exist_name: str) -> None:
        super().__init__(
            f"Card with name {name} already exists in registry; "
            f"Existing class name: {exist_name}"
        )


class CardKindMismatchError(ValueError):
    def __init__(self, cls_kind: str, value_kind: str) -> None:
        super().__init__(f"Static: {cls_kind} Dynamic: {value_kind}")


class InvalidCardKindError(ValueError):
    def __init__(self, value_kind: Any) -> None:
        super().__init__(f"value_kind has invalid type: {value_kind!r}")


class BaseCard(BaseModel, ABC):
    """The common base type for all cards in the game."""

    card_kind: ClassVar[str]

    _registries: ClassVar[dict[type, dict[str, type[BaseCard]]]] = {}

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return {"card_kind": self.card_kind}

    @classmethod
    def init_registry(cls, *, is_base: bool) -> None:
        if is_base:
            cls._registries[cls] = {}
            return

        if getattr(cls, "card_kind", None) is None:
            return

        for base_class in cls.__mro__:
            registry = cls._registries.get(base_class)
            if registry is None:
                continue

            if cls.card_kind not in registry:
                registry[cls.card_kind] = cls
                continue

            raise CardNameCollisionError(
                cls.card_kind, registry[cls.card_kind].__name__
            )

    @classmethod
    def __init_subclass__(
        cls,
        *,
        is_base: bool = False,
        **kwargs: Any,
    ) -> None:
        """Add a subclass to the Card registry."""
        super().__init_subclass__(**kwargs)

        cls.init_registry(is_base=is_base)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, BaseCard):
            return NotImplemented

        return self.card_kind < other.card_kind

    @staticmethod
    def _get_value(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, None)

        return getattr(obj, key, None)

    @classmethod
    def _validate_kind(cls, value_kind: Any, cls_kind: Any) -> str | None:
        if (
            value_kind is not None
            and cls_kind is not None
            and value_kind != cls_kind
        ):
            raise CardKindMismatchError(cls_kind, value_kind)

        result = value_kind or cls_kind
        if not isinstance(result, (str, type(None))):
            raise InvalidCardKindError(value_kind)

        return result

    @model_validator(mode="wrap")
    @classmethod
    def route_to_subclass(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> BaseCard:
        value_kind = cls._get_value(value, "card_kind")
        cls_kind = getattr(cls, "card_kind", None)

        kind = cls._validate_kind(value_kind, cls_kind)

        if kind is None:
            raise InvalidCardKindError(kind)

        registry = cls._registries.get(cls)
        concrete_type = registry.get(kind) if registry is not None else None
        if concrete_type is None or concrete_type is cls:
            return cast("BaseCard", handler(value))

        return concrete_type.model_validate(value)


BaseCard.init_registry(is_base=True)
