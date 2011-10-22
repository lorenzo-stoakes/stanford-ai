AI Notes
========

Unit 1 - Introduction
---------------------

## Course Overview ##

### Purpose of this class ###

* To teach you the basics of artificial intelligence

* To excite you

### Structure ###

Videos -> Quizzes -> Answer Videos

Assignments = quizzes without the answers :-) - graded.

Also, exams.

an AI program is called an 'intelligent agent'.

## Intelligent Agents ##

    Agent
    +----------------+   <- Sensors      Environment
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment
    |                |   Actuators ->    Environment
    +----------------+                   Environment

Happens in many iterations over the perception/action cycle.

## Applications of AI ##

### AI in Finance ###

    Trading Agent
    +----------------+   <- Rates/News   Environment  E.g. stock market
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment  Bonds
    |                |                   Environment
    |                |                   Environment  Commodities
    |                |   Trades ->       Environment
    +----------------+                   Environment

### AI in Robotics ###

    +----------------+   <- Cameras      Environment
    |                |      Microophone  Environment
    |                |      Touch        Environment
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment
    |                |     Motors  ->    Environment
    +----------------+     Voice         Environment

Initial web crawlers were called 'robot's, e.g. robots.txt.

We're going to focus to a large degree on robotics in the course.

### AI in Games ###

    Game Agent
    +----------------+   <- Your moves -  Environment
    |                |                    Environment
    |             |  |                    Environment
    |             |  |                    Environment E.g. you
    |             |  |                    Environment
    |             v  |                    Environment
    |                | - Its own moves -> Environment
    +----------------+                    Environment

### AI in Medicine ###

    Diagnostic Agent
    +----------------+    vital
    |                | <------------ you
    |                |   signals
    |                |
    |                |
    |                | diagnostics
    |                | ------------> doctor
    +----------------+

### AI and the Web ###

I.e. searching for 'great AI class'

    Crawler
    +----------------+   <- Web Pages
    |                |
    | DB             |                   World-wide-web
    |                |
    |                |
    |                |
    |                |
    +----------------+
       ^        |
       | Query  v Results

    <<<<----You---->>>>

## Terminology ##

* Fully- vs. Partially- Observable

Fully = what the agent can sense in the environment is *completely* sufficient for it to make an
optimal decision. E.g. card games where all the cards are the table + visible cards are sufficient
to make decisions.

Partially = Have to maintain a memory, e.g. poker. So not entirely sufficient.

    Agent
    +----------------+   <- Sensors      Environment  ---> state
    |                |                   Environment         |
    |                |                   Environment <--------
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment
    |                |   Actuators ->    Environment
    +----------------+                   Environment

Happens in many iterations over the perception/action cycle.

Fully observable if the sensors can see the entire state of the environment.
Partially observable if the sensors can't fully see state, keeps memory of previous state.

* Deterministic vs. stochastic environment

Deterministic where agent's actions uniquely determine the actions.

E.g. chess - outcome always the same for a given move.

Games like a dice-based game are stochastic - certain amount of randomness involved.

* Discrete vs. continuous

Discrete - Finitely (?! Can't understand what he said...) many action choices, and finitely many
things you can sense, e.g. chess - finitely many board positions, finitely many things you can do.

Continuous - Space of possible actions or things you can sense are infinite, e.g. throwing darts.

* Benign vs. adversarial (environments)

In benign environments, the environment might be random/stochastic, but it has no objective of its
own to contradict your own objectives. E.g. weather - it might be random, but it's not 'out to get
you'.

Adversarial, e.g. chess - where the opponent is actually 'out to get you'. It's much harder to find
good actions. In an adversarial situation, there is an agent which actively observes you +
counteracts what you are trying to achieve.

E.g. checkers - fully-observable, deterministic, discrete + adversarial

## Poker Question ##

E.g. poker - partially-observable, stochastic, discrete + adversarial

## Robot Car Question ##

E.g. robot car - partially-observable, stochastic, continuous, adversarial

Not adversarial apparently :) - quite subjective!

## AI and Uncertainty ##

AI as uncertainty management

AI = what to do when you don't know what to do?

Reasons for uncertainty:-

* Sensor limits
* Adversaries
* Stochastic environments
* Laziness
* Ignorance

## Machine Translation ##

E.g. google translate.

All done via AI/machine learning.

Done by taking lots of data, i.e. where there are already translations out there in language x and
language y.

If people ask for actual translation of article kept in records, then can just go and look that up ('we already have that').

Google also allow translation of individual words by keeping vast amounts of records of words in
different languages.

E.g. Chinese menu

Comparing rows with words common between them, e.g. 'wonton'. Also, the rows with 'wonton' in the
item name are also the only ones which contain a particular Chinese character - makes it highly
likely to be the character in question.

Looked up chicken - common char between Chicken entries.

## Chinese Translation 1 ##

Looking up a phrase - 2 chars common between two entries - 'corn cream'.

## Chinese Translation 2 ##

Lookup up a word again - soup - which os common between entries with 'soup' in them, well apart from
where it's a different character (?!). Correspondence doesn't have to be 100% for there to be a
correlation.

## Summary ##

Completed unit 1

* Key applications of AI
* Intelligent agent
* 4 key attributes
* Sources + management of uncertainty
* Rationality

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
    (s) -> { a_1, a_2, a_3, ... }
* Result
    (s, a) -> s'
* GoalTest
    (s) -> T|F
* PathCost (s-a->s... <- state-action transitions to number, n = cost of path) - most of the time
  the cost will be additive, i.e. the overall cost will be the individual steps' costs summed.
    (s-a->s-a->s)->n
* StepCost - components of the path cost.
    (s, a, s')->n

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

    f = g + h

    g(path) = path cost
    h(path) = h(s) = estimated distance to goal

Let's consider Romanian towns again. We use the heuristic h = straight line distance.

Whether A* works depends on the h function.

A* finds lowest cost path when

    h(s) <= true cost

We want h to:-

* never overestimate
* be optimistic
* be 'admissible' - i.e. admissable for the problem of finding the lowest cost path.

All essentially saying the same thing.

## Optimistic Heuristic ##

Optimistic h finds lowest-cost path

    f = g + h

    h(s) <= true cost s->g

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

    2 * 2 * 2 = 8

Consider results of moving left/right and turning on/off sucking as transitions between different
states.

Now consider the following additional factors:-

* Power switch  - on/off/sleep
* (Dust Camera) - on/off
* Brush height 1/2/3/4/5
* 10 positions

How many states?

    possible positions * dirty or not in each *

    10 * (2^10) * 3 * 2 * 5 = 307,200

Things can get big fast. This is why we need to use efficient algorithms.

## Sliding Blocks Puzzle + 1-2 ##

E.g.:-

    1  2  3  4
    5  6  7  8
    9  11 x  12
    13 10 14 15

Let's consider different parameters:-

    h_1 = # misplaced blocks = 4
    h_2 = sum(distances of blocks) = 4

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

    a block can move from A -> B
    if  (A adjacent to B)
    and (B is blank)

If we loosen the requirements + remove the B is blank criteria. This leaves us with our heuristic
from above, i.e.:-

    h_2 = sum(distances of blocks)

    a block can move from A -> B
    if  (A adjacent to B)

We could loosen further, and remove the criteria altogether:-

    a block can move from A -> B

Which leaves us with another heuristic:-

    h_1 = # misplaced blocks

So we can mechanically derive heuristics from the problem statement.

Another good heuristic is to take the maximum of the criteria heuristics, e.g.:-

    h = max(h_1, h_2)

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

    A -> S -> F

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

Unit 3 - Probability in AI
--------------------------

## Introduction ##

We're going to be looking at probabilities, especially structured probabilities, using Bayes
networks. The material is hard.

E.g. Bayes Network - car won't start, causes?

<img src="http://codegrunt.co.uk/images/ai/bayes1.png" />

This is a Bayes network - composed of nodes which correspond to events that you might/might not know
typically called 'random variables', linked by arcs where an arc indicates that the child is
influenced by its parent. Perhaps not in a deterministic way, can be linked in a probabilistic way,
e.g. battery age has a high chance of causing a dead battery, however it's not entirely necessary
(not all old batteries are dead).

There are 16 variables here. What the graph structure + associated probabilities imply is a huge
probability distribution. If we assume they are binary (as we will during this unit), then that's
2^16 possible values.

Once the network is set up, we can observe things like whether the lights are on, or whether the oil
light is on, and compute probabilities for the hypothesis, e.g. alternator broken, etc.

Going to look at how to construct these networks

* Binary events
* Probability
* Simple Bayes Networks
* Conditional Independence
* Bayes Networks
* D-Separation
* Parameter Counts
* Later: Inference in Bayes Networks

Very important.

Bayes used in almost all fields of smart computing, e.g. diagnostics, prediction, machine learning,
finance, google, robotics, particle filters HMM, MDP + POMDPs, Kalman Filters, etc. (we'll find out
about these odd-sounding applications later :-)

## Probability/Coin Flip + 2-5 ##

Cornerstone of AI. Used to express uncertainty, and the management of uncertainty is key in AI.

E.g. flipping a coin:-

    P(H) = 0.5, P(T) = 1 - 0.5 = 0.5

    P(H) = 0.25, P(T) = 1 - 0.25 = 0.75

    P(H, H, H) = 1/8 given P(H) = 0.5

Given:-

    x_i = result of i-th coin flip, where x_i = { H, T }

Then:-

    P(X_1 = X_2 = X_3 = X_4) = 1/(2^4) + 1/(2^4) = 1/8

Probability of 3 or more heads in 4 flips:-

    P({X_1, X_2, X_3, X_4} contains >=3 H) = 5/16 = 0.3125

Examined all the possibile outcomes shows that there are 5 possibilities:-

    HHHH *
    HHHT *
    HHTH *
    HHTT
    HTHH *
    HTHT
    HTTH
    HTTT
    THHH *
    THHT
    THTH
    THTT
    TTHH
    TTHT
    TTTH
    TTTT

## Probability Summary ##

    P(A) = p => P(notA) = 1 - p

Independence:-

    X|_Y: P(X)P(Y)   = P(X, Y)
          marginals    joint probability

## Dependence ##

<img src="http://codegrunt.co.uk/images/ai/dependence.png" />

## What We Learned ##

### Lessons ###

    P(Y) = Sigma{i}( P(Y|X=i) * P(X=i) )

    P(notX | Y) = 1 - P(X | Y)

    P(X | notY) = 1 - P(X | Y) <- NOT TRUE!

## Weather + 2-3 ##

    P(D_1 = sunny) = 0.9
    P(D_2 = sunny | D_1 = sunny) = 0.8
    P(D_2 = sunny | D_1 = rainy) = 0.6

    Since P(notX | Y) = 1 - P(X | Y):-
    P(D_2 = rainy | D_1 = sunny) = 1 - P(D_2 = sunny | D_1 = sunny) = 0.2

    Similarly:-
    P(D_2 = rainy | D_1 = rainy) = 1 - 0.6 = 0.4

By the Theory of Total Probability:-

    P(D_2 = sunny) = P(D_2 = sunny | D_1 = sunny) * P(D_1 = sunny) + P(D_2 = sunny | D_1 = rainy) * P(D_1 = rainy)

And:-

    P(D_3 = sunny) = P(D_3 = sunny | D_2 = sunny) * P(D_2 = sunny) + P(D_3 = sunny | D_2 = rainy) * P(D_2 = rainy)

Note that we assume the conditional probabilities are the same for D2->D3 as they are for D1->D2.

So:-

    P(D_2 = sunny) = 0.8 * 0.9 + 0.6 * (1 - 0.9) = 0.8 * 0.9 + 0.6 * 0.1 = 0.78
    P(D_3 = sunny) = 0.8 * 0.78 + 0.6 * (1 - 0.78) = 0.8 * 0.78 + 0.6 * 0.22 = 0.756

## Cancer + 2-4 ##

We can express the probability of cancer/not cancer thus;-

    P(C)    = 0.01
    P(notC) = 1 - 0.01 = 0.99

Let's say there's a test which comes out positive (+) or negative (-):-

    P(+|C) = 0.9
    P(-|C) = 1 - 0.9 = 0.1 since P(notA|B) = 1 - P(A|B).

    P(+|notC) = 0.2
    P(-|notC) = 0.8

We're after:-

    P(C|+)

But, first let's get some joint probabilities:-

<img src="http://codegrunt.co.uk/images/ai/cancer.png" />

Due to the rule of total probability:-

    P(+) = P(+|C)P(C) + P(+|notC)P(notC) = 0.9*0.01 + 0.2*0.99 = 0.207

We applied a sort of ad-hoc method to obtain the joint probabilities which, more formally, results
in:-

    P(A, B) = P(A) * P(B|A) = P(B) * P(A|B)

So:-

    P(B|A) = P(A, B)/P(A)
    P(A|B) = P(A, B)/P(B)

So
    P(C|+) = P(C, +)/P(+) = 0.009/0.207 =~ 0.0435

## Bayes Rule ##

Invented by Rev. Thomas Bayes, mathematician.

    P(A|B) = P(B|A)*P(A)/P(B)

Where:-

    P(A|B) = Posterior
    P(B|A) = Likelihood
    P(A)   = Prior
    P(B)   = Marginal Likelihood

Let's say that B is the evidence, and A is what we're interested in, e.g. test result vs cancer.

This is 'diagnostic reasoning' - given evidence, looking at the cause, i.e. - given something
observable, what is the probability of the non-observable thing?

Bayes turns this upside down to 'causal reasoning':-

>'Given, hypothetically, we knew the cause, what would be the probability of the evidence we just
> observed?'

To correct for this inversion, we have to multiply by the prior of the cause to be the case in the
first place, and divide it by the probability of the evidence, which is often expanded to:-

    P(A|B) = P(B|A)*P(A)/Sigma{a}(P(B|A=a)P(A=a))

Using total probability.

    P(C|+) = P(+|C) * P(C) / P(+) = P(+|C) * P(C) / (P(+|C)*P(C) + P(+|C)*P(notC))
           = 0.9 * 0.01 / (0.9 * 0.01 + 0.2 * 0.99)
           =~ 0.435

Which is the same as above :)

## Bayes Network ##

We can represent the kind of reasoning we performed in the above example graphically:-

<img src="http://codegrunt.co.uk/images/ai/bayes2.png" />

The information on the right-hand column is what we have, 

## Computing Bayes Rule ##

## Two Test Cancer + 2 ##

## Conditional Independence + 2 ##

## Absolute and Conditional ##

## Confounding Cause ##

## Explaining Away + 2-3 ##

## Conditional Dependence ##

## General Bayes Net + 2-3 ##

## Value of a Network ##

## D-Separation + 2-3 ##

## Congratulations! ##

Unit 4 - Probabilistic Inference
--------------------------------
