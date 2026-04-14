- **Add influence cap/decay**
- **Add coop win condition**
- Improve first strike mechanic?
- Add a TUI interface.
- Remove mypy after we have more experience with ty
- Make OperationDefinitions serializable and load them from a file.
- Use ConfigDict(frozen=true) in most pydantic models.
- Use ConfigDict(frozen=true) in GameState (requires a big refactor)
- ProxyWarCrisis, NuclearMeltdownCrisis and RogueProliferationCrisis
  need to be reviewed to ensure they provide sufficient player
  instruction and description of its resolution effects and
  mechanics when there is insufficient combined player influence
  to meet the initial threshold.
