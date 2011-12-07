1. Agents and Environments
--------------------------

Hints:

"There may be environments where the payoff is completely independent of the actions of the agent".

"There exists (at least) one environment..." means you can come up with any environment you want in order to try and make the statement true (e.g. defining what actions do, if anything, what the goals/rewards are, etc)."

True/False?

* There exists (at least) one environment in which every agent is rational.

False

Justification: a 'rational agent' is defined by always doing the right thing. However, in a given
environment, you might have two agents which do the opposite thing, and one assumes only one of the
actions is 'right'. Therefore it can't exist.

* For every agent, there exists (at least) one environment in which the agent is rational.

True

Justification: one could construct an environment in which the agent's actions are always correct.

* To solve the siding-tile 15-puzzle, an optimal agent that searches will usually (i.e. not in
  extreme edge-cases) require less memory than an optimal table-lookup reflex agent.

True

Justification: There are a lot of states, so to search effectively involves exploring a lot of
lines, however a reflex lookup table will have to store every possible state of the 15 squares.

* To solve the sliding-tile 15-puzzle, an agent that searches will always do better (=find shorter
  paths) than a table-lookup reflex agent.

False

Justification: If, for reflex, we have stored the correct action for every case, then our play will
be perfect + thus search doesn't always do better.

2. A* Search
------------

For heuristic function h, and action cost 10 (per step), enter, per node, order when nodes are
expanded (= removed from queue). Start with 1 at start state. Enter 0 if node will never be
expanded. Is the heuristic h admissible?

Note that a heuristic is admissible if it is LESS THAN OR EQUAL to the costs.

Not admissible as final step more expensive than real path.

3. 
