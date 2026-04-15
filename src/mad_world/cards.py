"""Basic infrastructure for polymorphic Card classes."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, cast

from pydantic import (
    BaseModel,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    model_serializer,
    model_validator,
)


class CardNameCollisionError(Exception):
    def __init__(self, name: str, exist_name: str) -> None:
        super().__init__(
            f'Card with kind "{name}" already exists in registry; '
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

    _registry: ClassVar[dict[str, type[BaseCard]]] = {}

    @model_serializer(mode="wrap")
    def serialize(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        result = cast("dict[str, Any]", handler(self))
        result["card_kind"] = self.card_kind
        return result

    @classmethod
    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        """Add a subclass to the Card registry."""
        super().__init_subclass__(*args, **kwargs)
        kind = getattr(cls, "card_kind", None)

        if kind is None:
            return

        if kind in cls._registry:
            raise CardNameCollisionError(kind, cls._registry[kind].__name__)

        cls._registry[kind] = cls

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, BaseCard):
            return NotImplemented

        return self.card_kind < other.card_kind

    @staticmethod
    def _get_kind(obj: Any) -> str | None:
        result = (
            obj.get("card_kind", None)
            if isinstance(obj, dict)
            else getattr(obj, "card_kind", None)
        )
        return result if isinstance(result, (str, type(None))) else None

    @model_validator(mode="wrap")
    @classmethod
    def route_to_subclass(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> BaseCard:
        static_kind = BaseCard._get_kind(cls)
        dynamic_kind = BaseCard._get_kind(value)

        static_cls = cls._registry.get(static_kind or "", None)
        dynamic_cls = cls._registry.get(dynamic_kind or "", None)

        if (
            dynamic_cls is not None
            and static_cls is not None
            and dynamic_cls != static_cls
        ):
            raise CardKindMismatchError(
                static_cls.__name__, dynamic_cls.__name__
            )

        final_cls = dynamic_cls or static_cls
        if final_cls == cls:
            return cast("BaseCard", handler(value))

        if final_cls and cls not in final_cls.__mro__:
            raise CardKindMismatchError(cls.__name__, final_cls.__name__)

        if final_cls is not None:
            return final_cls.model_validate(value)

        raise InvalidCardKindError(dynamic_kind or static_kind)
