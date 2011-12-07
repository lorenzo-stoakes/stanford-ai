AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 2 - Problem Solving
------------------------

## Introduction ##

Theory and technology of building agents which can plan ahead to solve problems.

Complexity of problem comes from there being many states, e.g. many signs on signposts, partial
observability in fog.

## What is a Problem? ##

Can't drive to a location not shown on the map. But once the location is shown, then yes.

### Definition of a Problem ###

* Initial state
* Actions (s is state)

    [; (s) \to { a_1, a_2, a_3, ... } ;]

* Result

    [; (s, a) \to s' ;]

* GoalTest

    [; (s) \to T|F ;]

* PathCost (s-a->s... <- state-action transitions to number, n = cost of path) - most of the time
  the cost will be additive, i.e. the overall cost will be the individual steps' costs summed.

    [; (s-a \to s-a \to s) \to n ;]

* StepCost - components of the path cost.

    [; (s, a, s') \to n ;]

## Example: Route Finding ##

Set of all states = state space

We navigate the state space by applying actions, specific to each city, e.g. which road to follow.

'Being in Arad' is path of length 0.

At every point we want to separate out the path into different subsets:-

* Frontier - Points at the farthest edge of the explored subset.
* Explored - Points we've already found.
* Unexplored  - Points we've yet to explore.

Path cost = sum of step costs.

## Tree Search ##

Superimposes search tree over state space.

Define a function:-

    function TREE-SEARCH(problem):
        frontier = {[initial]}
        loop:
            if frontier is empty: return FAIL
            path = remove-choice(frontier)
            s = path.end
            if s is a goal: return path
            for a in actions:
                add [path + a -> Result(s, a)] to frontier

remove-choice is where things vary - this is where the choice of algorithm has an impact.

### Breadth-First Search ###

Could be called 'shortest first search' - always chooses from the frontier a path which hasn't been
considered yet, which is the shortest possible path.

See video for the map graph + search tree.

Can remove regressive steps.

## Graph Search ##

    function GRAPH-SEARCH(problem):
        frontier = {[initial]}; explored = {}
        loop:
            if frontier is empty: return FAIL
            path = remove-choice(frontier)
            s = path.end; add s to explored
            if s is a goal: return path
            foor a in actions:
                add [path + a -> Result(s, a)] to frontier
                unless Result(s, a) in frontier + explored

## Uniform Cost Search + 1-5 ##

Could be called 'cheapest-first' - guaranteed to find the path with cheapest cost.

## Search Comparison + 1-3 ##

Compare -

Breadth-first  - Expand first the shallowest paths
Cheapest-first - Expand first the cheapest paths (lowest total costs)
Depth-first    - Expand first the longest paths

Breadth- + Cheapest- First search optimal, depth- first *not*.

Breadth-first optimal because we're always expanding the shortest paths first.

Cheapest-first optimal because we're always expanding the cheapest paths first.

Given the non-optimality of depth-first search, why would anyone use it? The answer is - storage
space.

Consider a very large (or possibly infinite) binary tree.

The frontier of breadth-first search will be a horizontal line of 2^n nodes.

Cheapest-first will be a non-obvious, possibly squiggly line.

Depth-first will have a frontier of n nodes, rather than 2^n nodes, which is a substantial saving.

If we keep track of the explored set, then we don't get this saving, however without it, we make a
significant saving.

Another issue is completeness.

Breadth- and Cheapest- first searches are complete - we'll eventually get there for e.g. infinite
depth trees (assuming finite cost in cheapest-first search). In depth-first, an infinite path will just keep on going and never find the goal.

## More on Uniform Cost ##

Start at start state, then gradually expand out (see video for graphic :-) - concentric circles).

However, we are essentially looking at all possibilities of a certain cost, rather than weighted
towards where the goal might actually be. On average, we're going to have to explore half of all the
possibilities before finding the goal. Unfortunately given what we know, we can't do much better
than that.

What's proved of most use is an estimate of the distance between a state and the goal.

A useful algorithm for this is 'greedy best-first search'.

Similar to contours of uniform cost approach, only weighted towards goal.

If you place an impenetrable barrier between the state and the goal, which permits a long path
which, though long, gets closer + closer to the goal, and also a short path, but which involves
briefly moving further from the goal, then greedy best-first does poorly.

## A* Search + 1-5 ##

Always expands the path with the minimum value of function f, which is defined as:-

    [; f = g + h ;]

    [; g(path) = ;] path cost
    [; h(path) = h(s) = ;] estimated distance to goal

Let's consider Romanian towns again. We use the heuristic h = straight line distance.

Whether A* works depends on the h function.

A* finds lowest cost path when

    [; h(s) \leq ;] true cost

We want h to:-

* Never overestimate.
* Be optimistic.
* Be 'admissible' - i.e. admissable for the problem of finding the lowest cost path.

All essentially saying the same thing.

## Optimistic Heuristic ##

Optimistic h finds lowest-cost path

    [; f = g + h ;]

    [; h(s) \leq ;] true cost [; s \to g ;]

At the goal, f = g + h = g + 0 = the actual cost.

All the paths on the frontier have estimated cost > c since we've explored the points
cheapest-first.

If h is optimistic, then the estimated cost is less than the true cost.

The path selected must then have an estimated cost less than the true cost of any of the paths on
the frontier.

## State Spaces + 1-3 ##

We've looked at the state space of a physical realm, i.e. a map of Romania.

We can look at other state spaces too + deal with abstract properties. Consider a vacuuming robot.

There are two positions in the 'vacuum world'. How many states are there?

2 - possible positions
2 - dirty/not? in each.

    [; 2 \times 2 \times 2 = 8 ;]

Consider results of moving left/right and turning on/off sucking as transitions between different
states.

Now consider the following additional factors:-

* Power switch  - on/off/sleep
* (Dust Camera) - on/off
* Brush height 1/2/3/4/5
* 10 positions

How many states?

    possible positions   [; \times ;]
    dirty or not in each [; \times ;]
    power switch states  [; \times ;]
    dust camera states   [; \times ;]
    brush height states

    = [; 10 \times (2^{10}) \times 3 \times 2 \times 5 = 307,200 ;]

Things can get big fast. This is why we need to use efficient algorithms.

## Sliding Blocks Puzzle + 1-2 ##

E.g.:-

    1  2  3  4
    5  6  7  8
    9  11 x  12
    13 10 14 15

Let's consider different parameters:-

    [; h_1 = ;] # misplaced blocks [; = 4 ;]
    [; h_2 = \sum ( ;]distances of blocks [; ) = 4 ;]

    10 would have to move 1 space
    11 would have to move 1 space
    14 would have to move 1 space
    15 would have to move 1 space

Which are admissable? Both. As each tile must be moved at least once, so will never be
'pessimistic'.

### Where is the Intelligence? ###

We're trying to build an AI to solve problems like this on its own.

Search algorithms to do a good job, but have to provide it with a heuristic function from 'the
outside' - the intelligence comes from the humans programming it.

Can we automatically come up with good heuristic functions?

Let's feed it the block puzzle:-

    a block can move from A [; \to ;] B
    if  (A adjacent to B)
    and (B is blank)

If we loosen the requirements + remove the B is blank criteria. This leaves us with our heuristic
from above, i.e.:-

    [; h_2 = \sum( ;] distances of blocks [; ) ;]

    a block can move from A [; \to ;] B
    if  (A adjacent to B)

We could loosen further, and remove the criteria altogether:-

    a block can move from A [; \to ;] B

Which leaves us with another heuristic:-

    [; h_1 = ;] # misplaced blocks

So we can mechanically derive heuristics from the problem statement.

Another good heuristic is to take the maximum of the criteria heuristics, e.g.:-

    [; h = max(h_1, h_2) ;]

Since both heuristics which are acting as parameters here are admissible, then the new heuristic
will be admissible too.

There could be a cost around computing these heuristics.

'Crossing out' parts of the rules like this is called 'generating a relaxed problem'. We've taken a
hard problem and made it easier.

We've essentially been adding new links in a state space, which makes the problem easier.

## Problems with Search ##

Consider the Romanian map problem - we essentially follow the algorithm without considering whether
or not things have *gone wrong* along the way.

Problem solving works when:-

* Fully observable - Domain is fully observable - must be able to see what state we start with.
* Known - Have to know set of available actions to us.
* Discrete - Must be finite number of actions to choose from.
* Deterministic - Have to know the result of taking an action.
* Static - Must be nothing else which can change the world except for our own actions.

Later we will find how to deal with these conditions not being true.

## A Note on Implementation ##

We've talked about paths in the state space, e.g.:-

    [; A \to S \to F ;]

How do we implement this in a computer program?

    +---------------+ +---------------+ +---------------+
    |     State A   | |    State S    | |    State F    |
    |               | |               | | Action SF     |
    |               | |               | | Cost 239      |
    |parent 0     <-----parent      <-----Parent        |
    |               | |               | |               |
    |               | |               | |               |
    +---------------+ +---------------+ +---------------+

4 fields -

* State - State at the end of the path.
* Action - Action taken to get there.
* Cost - Total cost.
* Parent - Pointer to other node, e.g. F has parent node S.

Linked-list of nodes representing the path.

Path = abstract idea.
Node = representation in computer memory.

Effectively synonyms because in one-to-one correspondence.

We have two lists:-

* Frontier
* Explored

In the frontier, we need to remove the best item + add in new items.

This implies we should use a priority queue.

We also need a membership test - is a new item in the priority queue? Which implies we should use a
set. So implemented as both in efficient programs.

Explored set can simply be represented as a set (e.g. hashtable or a tree) (funnily enough! :)
