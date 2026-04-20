1. **Analyze Potential New Crises**
   - **Idea 1: Space Race**
     - *Mechanics:* Players bid GDP for a large Influence reward for the highest bidder. If the combined bid exceeds a threshold, both players lose GDP and the Doomsday Clock advances.
     - *Analysis:* Similar to the AI Arms Race, but replaces the world-ending threshold with a severe mutual penalty. Fits the Cold War theme, but overlaps heavily with existing bidding mechanics without introducing fundamentally new interaction types.
   - **Idea 2: Humanitarian Disaster / Famine Relief**
     - *Mechanics:* A cooperative threshold crisis requiring a combined GDP bid. Failure does not end the world (unlike Chernobyl or Proxy War), but advances the clock significantly and drains Influence from both players.
     - *Analysis:* Introduces a cooperative crisis where failure is not an immediate game over. This allows players to intentionally fail if they think the opponent will suffer more from the fallout, adding a new strategic dimension.
   - **Idea 3: Cyber Sabotage**
     - *Mechanics:* Players bid Influence to breach each other's networks. The highest bidder steals 10 GDP directly from the lowest bidder. However, if the combined Influence bid exceeds 15, the aggressive malware escapes containment, causing both players to lose 15 GDP and the Doomsday Clock to advance by 10.
     - *Analysis:* Introduces a direct resource transfer (stealing GDP) which is a new mechanic not seen in other crises. The non-lethal combined threshold punishes over-aggression by hurting economies and advancing the clock, rather than ending the game. It creates a fascinating dynamic where players must balance the desire to steal (or protect their GDP) against the risk of triggering a global infrastructure collapse.

   *Selection:* **Cyber Sabotage** is the best idea. It introduces the novel mechanic of direct resource theft and a non-world-ending threshold penalty, adding fresh strategic depth and fitting perfectly with the game's espionage/modern-warfare themes.

2. **Implement `CyberSabotageCrisis` and `CyberSabotageAction`**
   - In `src/mad_world/crises.py`, add the `CyberSabotageAction` extending `BaseAction`.
   - Add `CyberSabotageDefs` containing the constants (`STEAL_GDP = 10`, `INF_THRESHOLD = 15`, `BACKLASH_GDP = -15`, `BACKLASH_CLOCK = 10`).
   - Create `CyberSabotageCrisis` extending `GenericCrisis[CyberSabotageAction]`.
   - Implement `get_default_action` and `resolve` methods.
   - Append `CyberSabotageCrisis` to `INITIAL_CRISIS_DECK`.

3. **Add Unit Tests for `CyberSabotageCrisis`**
   - In `tests/test_crises.py`, add `TestCyberSabotageCrisis` extending `CrisisTestBase`.
   - Test outcomes: player 1 wins, player 2 wins, tie, and the threshold exceeded scenario.

4. **Complete pre-commit steps**
   - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.

5. **Submit the changes**
   - Submit the PR with the analysis included in the description.
