# Mad World

A small computer strategy game built with modern Python.

## Game Overview

Mad World is a two-player geopolitical strategy game where the primary goal is
to achieve economic superiority (GDP) over your opponent without escalating
global tensions to the point of nuclear annihilation.

The game revolves around two key metrics:
- **GDP (Gross Domestic Product):** Your score. The player with the highest GDP
  at the end of the game wins.
- **Influence:** The currency you use to perform operations in the world.

### The Doomsday Clock
Every aggressive action, desperate bid for influence, or unmanaged crisis adds
escalation debt to the global stage, represented by colored cubes on the
Escalation Track. The number of cubes on the track represents the **Doomsday
Clock**. If the Doomsday Clock reaches its maximum limit (usually 30), the world
is destroyed, and nobody wins.

## Game Mechanics

A typical game runs for 10 rounds, with each round divided into several phases:

1. **Round Events:** A random event card is drawn, applying global effects,
   altering player stats, or pushing the Doomsday Clock closer to midnight.
2. **Bidding:** Players bid secretly for additional Influence. The higher your
   bid, the more Influence you gain, but high bids also increase the Doomsday
   Clock. Submitting a bid of 0 de-escalates the clock slightly.
3. **Operations:** Players spend their Influence to take actions. Operations can
   increase your GDP, decrease your opponent's GDP, drain their Influence, or
   push the Doomsday Clock. Taking no action or de-escalating operations can
   sometimes lower global tensions.
4. **Crises:** Severe global emergencies may arise, forcing players to respond.

Players can communicate with each other through limited messages during the
game, which can be useful when playing against AI personas.

## Installation and Running

### Prerequisites
- Python 3.12+
- `uv` (Fast Python package installer and resolver)
- [Optional] Ollama (if playing against LLM-based players running locally)

### Setup

```bash
make setup
```

### Running the Game

To launch the game with the default configuration:

```bash
uv run mad_world
```

You can view the full list of options by running:

```bash
uv run mad_world --help
```

To run a game between two AI agents, you can specify configurations for `alpha`
(Player 1) and `omega` (Player 2). For example, to have an LLM play against a
trivial bot:

```bash
uv run mad_world \
    --alpha '{"kind": "llm", "name": "LLM", "model": "gemma3:12b"}' \
    --omega '{"kind": "trivial", "name": "Bot", "bot_name": "Pacifist"}'
```

## Development

The project uses a standard set of modern Python tools:
- **ruff**: For fast linting and code formatting.
- **pytest**: For writing and executing tests.

### Running Checks

The project enforces a strict **100% test coverage**. To run the linters, type
checker, and tests against the current state of the code:

```bash
make all-checks
```

### Formatting

To format your code:

```bash
make format
```

### Testing

To run the test suite:

```bash
make test
```
