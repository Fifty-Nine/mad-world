This pull request modifies `BaseCard.serialize` to correctly serialize subclasses.

Specifically, it replaces `handler(self)` with `dict(self)` inside the `BaseCard`
`@model_serializer(mode="wrap")` function. Because `handler(self)` depends on the
declared schema (which belongs to `BaseCard` and knows nothing about dynamic
subclass attributes), it drops all custom fields defined by subclasses. By using
`dict(self)`, we accurately capture the full set of attributes on the instantiated
subclass, while preserving the manually-injected `card_kind` discriminator.
