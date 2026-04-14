- **Rename GamePlayer.name**
  - Consider renaming `GamePlayer.name` to better reflect it is the country name rather than the player persona name.
- **Add influence cap/decay**
- **Add coop win condition**
- Improve first strike mechanic?
- Add a TUI interface.
- Remove mypy after we have more experience with ty
- Make OperationDefinitions serializable and load them from a file.
- Use ConfigDict(frozen=true) in most pydantic models.
- Use ConfigDict(frozen=true) in GameState (requires a big refactor)
- Create a fluent interface/DSL for defining and manipulating game loop callbacks (similar to EventStream).
- ProxyWarCrisis, NuclearMeltdownCrisis and RogueProliferationCrisis
  need to be reviewed to ensure they provide sufficient player
  instruction and description of its resolution effects and
  mechanics when there is insufficient combined player influence
  to meet the initial threshold.
- Add a "model debug" mode that converts (repeated) model generation failures
  into hard errors (rather than silently accepting them).
- Add a way to easily save fun model personas for later.
