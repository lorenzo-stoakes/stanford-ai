
AI Notes
========

N.B. - I had to give up on attempting ascii art for everything. Turned out not to be all that
practical! I have started scanning notes for stuff I have to write. Other than that, am using
[tex the world](http://thewe.net/tex/) for equations (not all, yet, am working on it).

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

An AI program is called an 'intelligent agent'.

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

Unit 3 - Probability in AI
--------------------------

## Introduction ##

We're going to be looking at probabilities, especially structured probabilities, using Bayes
networks. The material is hard.

E.g. Bayes Network - car won't start, causes?

<img src="http://codegrunt.co.uk/images/ai/3-introduction-1.png" />

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

    [; P(H) = 0.5, P(T) = 1 - 0.5 = 0.5 ;]

    [; P(H) = 0.25, P(T) = 1 - 0.25 = 0.75 ;]

    [; P(H, H, H) = 1/8 given P(H) = 0.5 ;]

Given:-

    [; x_i = ;] result of i-th coin flip, where [; x_i = { H, T } ;]

Then:-

    [; P(X_1 = X_2 = X_3 = X_4) = 1/(2^4) + 1/(2^4) = 1/8 ;]

Probability of 3 or more heads in 4 flips:-

    [; P({X_1, X_2, X_3, X_4} contains /geq 3 H) = 5/16 = 0.3125 ;]

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

    [; P(A) = p => P(notA) = 1 - p ;]

Independence:-

    [; X \perp Y: P(X)P(Y)   = P(X, Y) ;]
                 marginals    joint probability

## Dependence ##

<img src="http://codegrunt.co.uk/images/ai/3-dependence-1.png" />

## What We Learned ##

### Lessons ###

    [; P(Y) = \sum_i(P(Y|X=i)P(X=i)) ;]

    [; P(\lnot X | Y) = 1 - P(X | Y) ;]

    [; P(X | \lnot Y) = 1 - P(X | Y) ;] <- NOT TRUE!

## Weather + 2-3 ##

    [; P(D_1 = sunny) = 0.9 ;]
    [; P(D_2 = sunny | D_1 = sunny) = 0.8 ;]
    [; P(D_2 = sunny | D_1 = rainy) = 0.6 ;]

Since

    [; P(notX | Y) = 1 - P(X | Y) ;] :-
    [; P(D_2 = rainy | D_1 = sunny) = 1 - P(D_2 = sunny | D_1 = sunny) = 0.2 ;]

Similarly:-

    [; P(D_2 = rainy | D_1 = rainy) = 1 - 0.6 = 0.4 ;]

By the Theory of Total Probability:-

    [; P(D_2 = sunny) = P(D_2 = sunny | D_1 = sunny)P(D_1 = sunny) + P(D_2 = sunny | D_1 = rainy)P(D_1 = rainy) ;]

And:-

    [; P(D_3 = sunny) = P(D_3 = sunny | D_2 = sunny)P(D_2 = sunny) + P(D_3 = sunny | D_2 = rainy)P(D_2 = rainy) ;]

Note that we assume the conditional probabilities are the same for D2->D3 as they are for D1->D2.

So:-

    [; P(D_2 = sunny) = 0.8 \times 0.9 + 0.6 \times (1 - 0.9) = 0.8 \times 0.9 + 0.6 \times 0.1 = 0.78 ;]
    [; P(D_3 = sunny) = 0.8 \times 0.78 + 0.6 \times (1 - 0.78) = 0.8 \times 0.78 + 0.6 \times 0.22 = 0.756 ;]

## Cancer + 2-4 ##

We can express the probability of cancer/not cancer thus;-

    [; P(C) = 0.01 ;]
    [; P(\lnot C) = 1 - 0.01 = 0.99 ;]

Let's say there's a test which comes out positive (+) or negative (-):-

    [; P(+|C) = 0.9 ;]
    [; P(-|C) = 1 - 0.9 = 0.1 ;]

Since

    [; P(\lnot A|B) = 1 - P(A|B) ;]

    [; P(+|\lnot C) = 0.2 ;]
    [; P(-|\lnot C) = 0.8 ;]

We're after:-

    [; P(C|+) ;]

But, first let's get some joint probabilities:-

<img src="http://codegrunt.co.uk/images/ai/cancer.png" />

Due to the rule of total probability:-

    [; P(+) = P(+|C)P(C) + P(+|\lnot C)P(\lnot C) = 0.9 \times 0.01 + 0.2 \times 0.99 = 0.207 ;]

We applied a sort of ad-hoc method to obtain the joint probabilities which, more formally, results
in:-

    [; P(A, B) = P(A)P(B|A) = P(B)P(A|B) ;]

So:-

    [; P(B|A) = \frac{P(A, B)}{P(A)} ;]
    [; P(A|B) = \frac{P(A, B)}{P(B)} ;]

So
    [; P(C|+) = \frac{P(C, +)}{P(+)} = \frac{0.009}{0.207} \simeq 0.0435 ;]

## Bayes Rule ##

Invented by Rev. Thomas Bayes, mathematician.

    [; P(A|B) = \frac{P(B|A)P(A)}{P(B)} ;]

Where:-

    [; P(A|B) = ;] Posterior
    [; P(B|A) = ;] Likelihood
    [; P(A) = ;] Prior
    [; P(B) = ;] Marginal Likelihood

Let's say that B is the evidence, and A is what we're interested in, e.g. test result vs cancer.

This is 'diagnostic reasoning' - given evidence, looking at the cause, i.e. - given something
observable, what is the probability of the non-observable thing?

Bayes turns this upside down to 'causal reasoning':-

>'Given, hypothetically, we knew the cause, what would be the probability of the evidence we just
> observed?'

To correct for this inversion, we have to multiply by the prior of the cause to be the case in the
first place, and divide it by the probability of the evidence, which is often expanded to:-

    [; P(A|B) = \frac{P(B|A)P(A)}{\sum_a P(B|A=a)P(A=a)} ;]

Using total probability.

    [; P(C|+) = \frac{P(+|C)P(C)}{P(+)} = \frac{P(+|C)P(C)}{P(+|C)*P(C)} + P(+|C)P(\lnot C) ;]
    [;       = 0.9 * 0.01 / (0.9 * 0.01 + 0.2 * 0.99) ;]
    [;       \simeq 0.0435 ;]

Which is the same as above :)

## Bayes Network ##

We can represent the kind of reasoning we performed in the above example graphically:-

<img src="http://codegrunt.co.uk/images/ai/3-bayes-network-1.png" />

The information on the right-hand column is what we have, and we want to perform diagnostic
reasoning, i.e. determining P(A|B) and P(A|notB).

Quiz - how many parameters:-

<img src="http://codegrunt.co.uk/images/ai/3-bayes-network-2.png" />

## Computing Bayes Rule ##

Looking at more complex networks.

Examining Bayes Rule again:-

    [; P(A|B) = \frac{P(B|A)P(A)}{P(B)} ;]

P(B|A) and P(A) are relatively easy to determine. P(B), not so much.

However, we can take a look at the negation of P(A|B), and cancel this term out:-

    [; P(\lnot A|B) = \frac{P(B|\lnot A)P(\lnot A)}{P(B)} ;]

We can simply ignore the 'normaliser' P(B), to give us P' 'psuedo-probability' terms:-

    [; P'(A|B) = P(B|A)P(A) ;]
    [; P'(\lnot A|B) = P(B|\lnot A)P(\lnot A) ;]

If we define a normaliser, eta, then we can get back to the actual probabilities thus:-

    [; P(A|B) = \eta P'(A|B) ;]
    [; P(\lnot A|B) = \eta P'(\lnot A|B) ;]

Where:-

    [; \eta = (P'(A|B) + P'(\lnot A|B))^{-1} ;]

We normalise the pseudo probabilities such that total probability holds:-

    [; P(A|B) + P(\lnot A|B) = 1 ;]

## Two Test Cancer + 2 ##

<img src="http://codegrunt.co.uk/images/ai/3-two-test-cancer-1.png" />

We declare a short-form:-

    [; P(C|T_1 = + T_2 = +) = P(C|++) ;]

So:-

    [; P(C|++) = \eta P'(C|++) ;]

And:-

    [; P'(C|+) = P(+|C)P(C) ;]

Assuming each test is independent of one another (conditionally independent, see below), then we
simply multiply by another P(+|C):-

Note: This seems extremely hand-wavy to me.

    [; P'(C|++) = P(+|C)P(+|C)P(C) = 0.9 \times 0.9 \times 0.01 = 0.0081 ;]
    [; P'(\lnot C|++) = P(+|\lnot C)P(+|\lnot C)P(\lnot C) = 0.2 \times 0.2 \times 0.99 = 0.0396 ;]

And to normalise:-

    [; \eta = (P'(C|++) + P'(\lnot C|++))^{-1} = 0.0081 \times 0.0396 \simeq 20.96 ;]

Thus:-

    [; P(C|++) = 20.96 \times 0.081 \simeq 0.1698 ;]
    [; P(\lnot C|++) = 20.96 \times 0.0396 \simeq 0.8302 ;]

Again, to determine P(C|+-), we follow a similar procedure:-

    [; P(C|+-) = \eta P'(C|+-) ;]
    [; P'(C|+-) = P(+|C)P(-|C)P(C) = 0.9 \times 0.1 \times 0.01 = 0.0009 ;]
    [; P'(\lnot C|+-) = P(+|\lnot C)P(-|\lnot C)P(\lnot C) = 0.2 \times 0.8 \times 0.99 = 0.1584 ;]
    [; \eta = (P'(C|+-) + P'(\lnot C|+-))^{-1} = (0.0009 + 0.1584)^{-1} \simeq 6.277 ;]

Thus:-

    [; P(C|+-) = 6.277 \times 0.0009 \simeq 0.00565 ;]
    [; P(\lnot C|+-) = 6.277 \times 0.1584 \simeq 0.994 ;]

## Conditional Independence + 2 ##

Introducing some terminology.

The 'hidden variable' C causes the stochastic test outcomes T1 and T2.

We didn't just assume that T1 and T2 are identically distributed, we also assumed that they were
*conditionally independent*.

If we knew with absolute certainty the value of C, it would tell us nothing to relate T1 to T2.

More formally:-

    [; P(T_2|C, T_1) = P(T_2|C) ;]

I.e. knowledge of T1 has absolutely no impact on T2.

This follows from the Bayes diagram - if we removed C from it, then T1 and T2 are essentially cut
off from one another.

Conditional independence is really important.

Looking at the following more general diagram:-

<img src="http://codegrunt.co.uk/images/ai/3-conditional-independence-1.png" />

Then we write:-

    [; Given A, B \perp C ;]

    [; B \perp C | A ;]

If we *don't* know the prior condition, i.e. A, then we *can't* say that B and C are
independent. This is because the result of B indicates something about the hidden A which will in
turn influence C. In our cancer case, one positive test result gives an indication as to whether we
have cancer, and thus another test result will be influenced by that too.

To drive the point home - let's calculate the probability of one test given the result of another:-

Let's use some short-hand once again:-

    [; P(T_1=+) = P(+_1), P(T_2=+) = P(+_2) ;]

We can use total probability to answer this:-

    [; P(+_2|+_1) = P(+_2|+_1, C)P(C|+_1) + P(+_2|+_1, \lnot C)P(\lnot C|+_1) ;]

Given conditional independence, this is equivalent to:-

    [; P(+_2|+_1) = P(+_2|C)P(C|+_1) + P(+_2|\lnot C)P(\lnot C|+_1) ;]

    [; P(+_2|+_1) = 0.9 \times 0.0435 + 0.2 \times 0.9565 \simeq 0.2305 ;]

## Absolute and Conditional ##

Let's look at the different forms of independence we've encountered:-

<img src="http://codegrunt.co.uk/images/ai/3-absolute-and-conditional-1.png" />

    [; A \perp B \not\Rightarrow A \perp B | C ;]

We will discuss why this is the case next :-)

    [; A \perp B | C \not\Rightarrow A \perp B ;]

As we've seen just now, conditional independence does not imply absolute independence since
something which affects one conditionally independent event can also affect the other.

## Confounding Cause ##

<img src="http://codegrunt.co.uk/images/ai/3-confounding-cause-1.png" />

    [; P(S)=0.7 ;]
    [; P(R)=0.01 ;]

    [; P(H|S, R) = 1 ;]
    [; P(H|\lnot S, R) = 0.9 ;]
    [; P(H|S, \lnot R) = 0.7 ;]
    [; P(H|\lnot S, \lnot R) = 0.1 ;]

This is a trick question. Since P(R) and P(S) are independent, P(R|S) = P(R) = 0.01!

## Explaining Away + 2-3 ##

(Again, working with the example given above)

Explaining away means - if we know that we are happy, then sunny weather can 'explain away' the
cause of happiness. If it's sunny, then it makes it less likely that there has been a raise.

If it's rainy, then it makes it more likely to be a raise since the happiness cannot be explained by
the weather.

If we see a certain effect which can be caused by multiple causes, then seeing one of those causes
can 'explain away' any other cause.

E.g., we want to determine:-

    [; P(R|H, S) ;]

We can use a sneaky trick here, by using a twist on Bayes:-

    [; P(A|B,C) = \frac{P(B|A,C)P(A|C)}{P(B|C)} ;]

I.e. - we still do the switch between A and B, only we take into account the fact that everything is
still predicated on C.

So:-

    [; P(R|H, S) = \frac{P(H|R, S)P(R|S)}{P(H|S)} = \frac{0.01}{P(H|S)};]

Since

    [; P(H|R, S) = 1 ;]
    [; P(R|S) = P(R) = 0.01 ;]

Carrying on:-

    [; P(H|S) = P(H|R, S)P(R|S) + P(H|\lnot R, S)P(\lnot R|S) ;]
    [; P(H|S) = 0.01 + 0.7 \times 0.99 = 0.703;]

Hence:-

    [; P(R|H, S) = \frac{0.01}{0.703} \simeq 0.0142 ;]

Let's determine P(R|H):-

    [; P(R|H) = \eta P'(R|H) ;]
    [; P'(R|H) = P(H|R)P(R) ;]
    [; P'(\lnot R|H) = P(H|\lnot R)P(\lnot R) ;]

    [; P(H|R) = P(H|R,S)P(S|R) + P(H|R,\lnot S)P(\lnot S|R) ;]

Again, since S and R are fully independent:-

    [; P(H|R) = P(H|R,S)P(S) + P(H|R,\lnot S)P(\lnot S) ;]
    [; P(H|R) = 1 \times 0.7 + 0.9 \times 0.3 = 0.97 ;]

Going through the same process for [; P(H|\lnot R) ;]:-

    [; P(H|\lnot R) = P(H|\lnot R, S)P(S|\lnot R) + P(H|\lnot R, \lnot S)P(\lnot S|\lnot R) ;]
    [; P(H|\lnot R) = 0.7 \times 0.7 + 0.1 \times 0.3 = 0.52 ;]

So now we can calculate the pseudo probabilities:-

    [; P'(R|H) = 0.97 \times 0.01 = 0.0097 ;]
    [; P'(\lnot R|H) = 0.52 \times 0.99 = 0.5148 ;]

And the normalisation factor:-

    [; \eta = (P'(R|H) + P'(\lnot R|H))^{-1} ;]
    [; \eta = (0.0097 + 0.5148)^{-1} \simeq 1.9066 ;]

Hence:-

    [; P(R|H) = 1.9066 \times 0.0097 \simeq 0.0185 ;]

The point here is that if he's happy but doesn't know about the weather, then the probability of a
raise is higher. The knowledge of the weather reduces the probability of the raise.

To calculate [; P(R|H, \lnot S) ;], we can (ab)use Bayes again:-

    [; P(R|H, \lnot S) = \frac{P(H|R, \lnot S)P(R|\lnot S)}{P(H|\lnot S)};]

And using the theorem of total probability one more:-

    [; P(H|\lnot S) = P(H| \lnot S, R)P(R|\lnot S) + P(H|\lnot S, \lnot R)P(\lnot R|\lnot S) ;]

Again, S and R are totally independent so:-

    [; P(H|\lnot S) = P(H| \lnot S, R)P(R) + P(H| \lnot S, \lnot R)P(\lnot R) ;]
    [; P(H|\lnot S) = 0.9 \times 0.01 + 0.1 \times 0.99 = 0.108;]

Also, let's take this into account for our original equation:-

    [; P(R|H, \lnot S) = \frac{P(H|R, \lnot S)P(R)}{P(H|\lnot S)} ;]

So, finally:-

    [; P(R|H, \lnot S) = \frac{0.9 \times 0.01}{0.108} \simeq 0.0833 ;]

## Conditional Dependence ##

It's interesting to compare all the outcomes regarding the raise:-

    [; P(R|S) = 0.01 ;]
    [; P(R|H, S) = 0.0142 ;]
    [; P(R|H, \lnot S) = 0.0833 ;]

H adds a dependence between S and R, despite them being independent.

<img src="http://codegrunt.co.uk/images/ai/3-conditional-dependence-1.png" />

Without information about H, the probability of R is completely unaffected by the knowledge of H.

    [; R \perp S ;]

However, when we know something about H, then things begin to get affected, i.e.:-

    [; P(R|H, S) = 0.0142 \not= P(R|H) ;]
    [; P(R|S) = 0.01 = P(R) ;]
    [; P(R|H, \lnot S) = 0.0833 \not= P(R|H) ;]

The probability of a raise, R, is affected by the probability of sunny weather.

This leads to the previously mentioned fact that full independence does not mean conditional
independence, i.e.:-

    [; R \perp S ;]
    [; R \not\perp S | H ;]

So, two variables that are independent might not be conditionally independent.

INDEPENDENCE DOES __NOT__ IMPLY CONDITIONAL INDEPENDENCE!

## General Bayes Net + 2-3 ##

We can now define Bayes networks in a more general way. Bayes networks define probability
distributions over a graph of random variables, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/3-general-bayes-net-1.png" />

Instead of enumerating all possibilities of all combinations of these 5 random variables, the Bayes
network is defined by probability distributions which is inherent to each individual node.

The probability at each node is only conditioned on the incoming arcs.

* A has no incoming arcs, so its probability is P(A).
* B has no incoming arcs, so its probability is P(B).
* C has incoming arcs from A and B, so its probability is P(C|A, B).
* D has an incoming arc from C, so its probability is P(D|C).
* E has an incoming artc from C, so its probability is P(E|C).

Thus:-

    [; P(A, B, C, D, E) = P(A).P(B).P(C|A, B).P(D|C).P(E|C) ;]

This has a big advantage in that the joint distribution of any five variables requires 2^5-1 = 31
probability values, whereas the Bayes network requires only 10, e.g.:-

2 From P(A), P(B):-

    [; P(A), P(B) =2 ;]

4 from P(C|A, B):-

    [; P(C|A, B), P(C|A, \lnot B), P(C|\lnot A, B), P(C|\lnot A, \lnot B) ;]

2 from P(D|C):-

    [; P(D|C), P(D|\lnot C) ;]

2 from P(E|C):-

    [; P(E|C), P(E|\lnot C) ;]

Scales a lot better to large networks than the combinatorial approach.

Some quizzes:-

<img src="http://codegrunt.co.uk/images/ai/3-general-bayes-net-2.png" />

<img src="http://codegrunt.co.uk/images/ai/3-general-bayes-net-3.png" />

## Value of a Network ##

And our original network:-

<img src="http://codegrunt.co.uk/images/ai/3-introduction-1.png" />

Which is:-

Row 1:-

    [; 1 + 1 + 1 = 3 ;]

Row 2:-

    [; 2 + 2^2 = 6 ;]

Row 3:-

    [; 2 + 2^2 + 1 + 1 + 1 + 1 = 10 ;]

Row 4:-

    [; 2 + 2^2 + 2^2 + 2^4 + 2 = 28 ;]

So:-

Total =

    [; 3 + 6 + 10 + 28 = 47 ;]

Which is quite an improvement on 65,535 using the combinatorial approach!

## D-Separation + 2-3 ##

<img src="http://codegrunt.co.uk/images/ai/3-d-separation-1.png" />

So:-

    [; C \perp A ;]

No, since A has an effect on C.

    [; C \perp A|B ;]

Yes, since once we assume B, we've taken into account A's effect.

    [; C \perp D ;]

No, since A effects C, which also effects D.

    [; C \perp D|A ;]

Yes, since A is the 'common ancestor' between C and D.

    [; E \perp C|D ;]

Yes, since D takes into account A's effect which is the common ancestor between C and E.

Put simply:-

Any two nodes are independent if they're not linked by just unknown variables. E.g., if we know B,
then anything downstream of B is independent of everything upstream of B.

<img src="http://codegrunt.co.uk/images/ai/3-d-separation-2.png" />

    [; A \perp E ;]

No, since A influences C which influences E.

    [; A \perp E | B ;]

No, since B doesn't exclude A's influence from affecting E.

    [; A \perp E | C ;]

Yes, since C does exclude A's influence on E.

    [; A \perp B ;]

Yup.

    [; A \perp B | C ;]

No, since C can explain things away.

This leads to the general study of conditional independence in Bayes networks, often calle D-separation or reachability.

D-separation is best studied by 'active tripets' and 'inactive triplets'.

* Active triplets render variables dependent.
* Inactive triplets render variables independent.

E.g.:-

<img src="http://codegrunt.co.uk/images/ai/3-d-separation-3.png" />

Final Quiz :-) :-

<img src="http://codegrunt.co.uk/images/ai/3-d-separation-4.png" />

Considering independence:-

    [; F \perp A ;]

Yes.

    [; F \perp A | D ;]

No, as D helps 'explain away' B and E, which then percolates up to A and F, hence they are *not*
independent.

    [; F \perp A | G ;]

Again, knowledge of G percolates up to do, which then 'explains away' B and E, which go back to A
and F, hence A and F are *not* independent.

    [; F \perp A | H ;]

H might tell us something about G, but it won't tell us anything about D, so A and F are not independent.

## Congratulations! ##

Learnt a lot:-

* Graph structure
* Compact Representation
* Conditional Independence

This was a largely theoretical unit. Will talk more about applications later.

Bayes networks are very useful for many applications.

Unit 4 - Probabilistic Inference
--------------------------------

## Overview and Example ##

In the previous unit we went over:-

* Probability Theory
* Bayes Net - A concise representation of a join probability distribution
* Independence
* Inference - How to answer probability questions using Bayes nets.

Let's look at a simple example:-

<img src="http://codegrunt.co.uk/images/ai/4-overview-and-example-1.png" />

Where B is a burglary and E an earthquake, A is an alarm J is Jill calling and M is Mike calling.

The usual question would be - what are the inputs and outputs. Instead of inputs we have 'evidence'
variables, instead of values we want we have 'query' variables. In addition, we have 'hidden'
variables, i.e. ones which are neither evidence nor query, but have to be computed internally.

In probabilistic inference, the output is not a single number for each of the queries, but rather a
probability distribution.

The answer is going to be a complete joint probability distribution over query variables, called the 'posterior distribution' given the evidence:-

Each evidence variable is given an exact value.

    [; p(Q_1, Q_2, ... | E_1 = e_1, E2 = e_2, ...) ;]

Which is the most likely explanation?

Which combination of values has the highest probability?

    [; argmax_q p(Q_1, Q_2, ... | E_1 = e_1, E2 = e_2) ;]

We don't need to go in just one direction, we can make the query variables be the evidence
variables, or vice-versa, or really any combination. Doesn't have to be causal.

Quiz- Mary has called to report the alarm is going off, and want to know whether there has been a
burglary.

M is evidence, B is query, the rest are hidden.

## Enumeration ##

This goes through the possibilities, adds them up, then finds the answer.

We use the definition of conditional probability:-

    [; P(Q|E) = \frac{P(Q, E)}{P(E)} ;]

Notation:-

    [; P(E=true) \equiv P(+E) ;]
    [; \equiv P(+e) ;]
    [; \equiv 1-P(\lnot e) ;]
    [; \equiv P(e) ;]

However we're sticking to use of + and [; \lnot ;] signs.

We want:-

    [; P(+b|+j, +m) = \frac{P(+b,+j,+m)}{P(+j,+m)} ;]

    [; P(+b, +j, +m) = \sum_e\sum_a P(+b, +j, +m, e, a) ;]
    [; = \sum_e\sum_a P(+b)P(e)P(a|+b, e)P(+j|a)P(+m | a) ;]

If we say that:-

    [; f(e, a) = P(+b)P(e)P(a | +b, e)P(+j | a)P(+m, a) ;]

Then:-

    [; = f(+e, +a) + f(+e, \lnot a) + f(\lnot e, +a) + f(\lnot e, \lnot a) ;]

Quiz - Plug in numbers.

## Speeding Up Enumeration + 2-4 ##

For a simple network, this is sufficient. Even if every variable was hidden, there would only be 32
rows to sum up ([; 2^5 = 32 ;]).

However, if we look at a larger network (e.g. a car insurance network)b, we'll run into
problems. E.g. one with 27 different variables, and if each was boolean, then we have
[; 2^27 \simeq 100M ;] rows to sum out. However, in fact there are some non-boolean variables (as
there are in this example network), then the numbers rapidly go up, here into the quadrillions.



### Pulling Out Terms ###

Let's look at the compound probability equation again:-

    [; \sum_e\sum_a P(+b)P(e)P(a|+b, e)P(+j|a)P(+m | a) ;]

We can take the [; P(+b) ;] term outside, meaning we only need to multiply by it once:-

    [; P(+b)\sum_e\sum_a P(+b)P(e)P(a|+b, e)P(+j|a)P(+m | a) ;]

We can move [; P(e) ;] to be outside the innermost nested summation, since it will be the same for
each e in the summation over e (i.e. it doesn't depend on a):-

    [; P(+b)\sum_e P(e) \sum_a P(+b)P(a|+b, e)P(+j|a)P(+m | a) ;]

Now we have 3 terms rather than 5, so the cost of each row in the table is less, but we still have
the same number of rows. We need to do better.

### Maximise independence ###

The structure of a Bayes net determines how efficient it is to perform inference on it.

E.g. a linear string of variables:-

<img src="http://codegrunt.co.uk/images/ai/4-speeding-up-enumeration-1.png" />

If all n variables are boolean variables.

In the alarm network shown earlier, we were careful to ensure that we put all the independence
relations in the structure of the network, but if we put the nodes in a different order we'd have a
different structure, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/4-speeding-up-enumeration-2.png" />

Explanation:-

If we just consider the J and M nodes from the previous network, then M is dependent on J due to the
fact that we are not aware whether the alarm has gone off, so knowledge of J influences knowledge of
M. Again the concept of 'dependence trumps independence' applies here.

If we then add the alarm A, then of course intuitively, both J and M influence A, as if John calls
then it's more likely that the alarm has gone off, the same goes for Mary. You can also go to the
conditional probability tables to determine whether the numbers work out.

If we add B, then it's clear that A influences B, but not J or M as J or M give us information on
the alarm which we already have.

If we then add E, both B and A influence E, as obviously whether the alarm is going off influences
whether there is an earthquake, but also the existence of the earthquake influences the likelihood
of it being a burglary, and vice-versa.

## Causal Direction ##

The moral - Bayes nets are the most compact and thus easiest to do inference on when they are
written in the causal direction, i.e. the networks flow from causes to effects.

## Variable Elimination + 2-4 ##

Let's look at our joint probability equation once again:-

    [; \sum_e \sum_a P(+b) P(e) P(a|+b, e) P(+j|a)P(+m, a) ;]

It's NP-hard to do inference over Bayes nets in general, however variable elimination works faster
than inference by enumeration in most practical cases.

This requires an algebra for manipulating factors which are multi-dimensional arrays of the
probabilistic variables contained in the above equation.

To explore this, let's look at a new network:-

<img src="http://codegrunt.co.uk/images/ai/4-variable-elimination-1.png" />

Where

* R - is it raining?
* T - is there traffic?
* L - late for next appointment?

    [; P(+l) = \sum_r \sum_t P(r)R(t|r)P(+l|t) ;]

The elimination:-

<img src="http://codegrunt.co.uk/images/ai/4-variable-elimination-2.png" />

The steps used to achieve elimination:-

1. Joining Factors.

E.g. combining the P(R) and P(T|R) tables here, making use of the definition of conditional probability:-

    [; P(T|R) = \frac{P(R, T)}{P(R)} ;]

Hence:-

    [; P(R, T) = P(T|R)P(R) ;]

Thus we end up with the following probability table:-

      P(R, T)

    +r, +t 0.08
    +r, -t 0.02
    -r, +t 0.09
    -r, -t 0.81

2. Summing out/ Marginalisation

Here we eliminate out a variable by simply summing joint probabilities such that you end up with a
marginal probability:-

    P(T)

    +t 0.17
    [; \lnot ;]t 0.83

Then we repeat the procedure to join again:-

    P(T, L)

    +t, +l 0.051
    +t, -l 0.119
    -t, +l 0.083
    -t, -l 0.747

Finally, we marginalise once again to obtain P(L):-

    P(L)
    +l 0.134
    -l 0.866

Continued process of joining together factors to get a larger factor, then eliminating variables by summing out.

Can be much more efficient than just enumeration, given this is done in sensible order.

## Approximate Inference ##

By means of sampling.

Let's say we want to deal with a joint probability distribution of heads + tails of 2 coins:-

E.g.:-

<img src="http://codegrunt.co.uk/images/ai/4-approximate-inference-1.png" />

Here we sample by simply running the experiment over and over and looking at the counts. If the
counts are low, then random variation can cause the results to be unreliable. However, as we add
more samples the counts we get come closer to the true distribution.

Sampling has an advantage over inference in that we have a procedure for coming up with at least an
approximate value for the joint probability distribution, rather than exact inference where the
computation might be v. complex.

Another advantage is that, if we don't know what the conditional probability tables are, we can
still proceed with sampling, unlike in exact inference where we simply couldn't.

## Sampling Example ##



## Approximate Inference 2 ##

## Rejection Sampling ##

## Likelihood Weighting + 1-2 ##

## Gibbs Sampling ##

## Monty Hall Problem ##

## Monty Hall Letter ##

Unit 5 - Machine Learning
-------------------------

## Introduction ##

* Lots of data in the world e.g. dna, finance, web, etc.

Machine learning is the discipline of extracting information from this data.

## What is Machine Learning ##

We've already talked about Bayes networks.
Machine learning is about finding these networks based on data.
Machine learning = learn models from data.

E.g. google/amazon/etc.

We'll start with supervised learning then moving on to unsupervised learning.

## Stanley DARPA Grand Challenge ##

Stanley is a self-driving car - no human assistance whatsoever.

Uses a laser system to build models of the terrain. Problem is, they only see 25m. If you drive fast
you have to see further.

Uses machine learning principles on the laser-scanned portion of the road in conjunction with the
camera view in order to see right up to the horizon.

Means you can drive fast.

Key factor in winning the race.

## Taxonomy ##

Very large field

Very basic terminology:-

### What? ###

* Parameters - e.g. probabilities of a Bayes network.
* Structure - e.g. Arc structure of a Bayes network.
* Hidden concepts - e.g. Certain training example might form a group. Can help you make better sense
  of the data.

### What From? ###

Machine learning driven by some sort of information that you care about:-

* Supervised learning - We use specific target labels.
* Unsupervised learning - target labels are missing and we use replacement principles to find, for
  example, hidden concepts.
* Reinforcement learning - An agent learns from feedback from a physical environment, where it
  receives some sort of response to actions, e.g. 'well done' or 'that works'.

### What For? ###

* Prediction - e.g. stock market
* Diagnostics - e.g. medicine
* Summarisation - e.g. summarising a long article

### How ###

* Passive - If the agent has no impact on the data itself.
* Active - The agent has some impact.
* Online - Data obtained while being generated.
* Offline - Data obtained after been generated.

### Outputs? ###

* Classification - Output is binary/fixed number of classes e.g. chair or not
* Regression - Output is continuous, e.g. 66.5 degrees Celcius

### Details? ###

* Generative - Seek to model the data as generally as possible.
* Discriminative - Seek to distinguish data.

Might seem like a trivial distinction, however it makes a huge difference in the implementation of
the algorithm.

### Supervised Learning ###

Most of the work in the field is supervised learning.

For each training example given a feature vector and a target label called y:-

    [; x_1 x_2 x_3 ... x_n \to y ;]

e.g. for credit rating - feature vector = person's salary, whether they've defaulted before, etc.,
target label = credit rating.

Based on previous data and actual instances of default. The idea is to predict future customer
outcomes, e.g. person comes in with a different feature vector - are they likely to default or not?

Can apply the exact same idea to image recognition, e.g. x's are pixels, and y is whether or not recognised:-

    [; \begin{bmatrix}
        x_{11} & x_{12} & x_{13} & ... & x_{1n} & \to & y_1 \\
        x_{21} & x_{22} & x_{23} & ... & x_{2n} & \to & y_2 \\
        ...    & ...    & ...    & ... & ...    & ... & ... \\
        x_{m1} & x_{m2} & x_{m3} & ... & x_{mn} & \to & y_m \\
       \end{bmatrix} ;]

We call this the data. For each vector

    [; x_m ;]

We want to find a function as follows:-

    [; f(x_m) = y_m ;]

This isn't always possible, so sometimes it's tolerable to permit a certain amount of error.

We can then use this function for future x's to obtain accurate y's.

## Occam's Razor ##

<img src="http://codegrunt.co.uk/images/ai/5-occams-razor-1.png" />

If you compare a. to b., you can see that both fit the data perfectly, however b is clearly less
appropriate as it violently oscillates all over the place and it is unlikely (let's face it -
impossible) that it will actually fit the data between these points.

> Everything else being equal, choose the less complex hypothesis

There is a trade-off:-

    Fit <--------------------> Low Complexity

E.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-occams-razor-2.png" />

Here complexity could be polynomial degree. As complexity increases, training data error reduces as
you fit the data more accurately, however overall error (generalisation error) increases, which is
the sum of training data error and 'overfitting' error.

The best complexity is obtained is where the generalisation error is at a minimum.

There are methods for determining the overfitting error, in a field called 'Bayes variance methods'.

Often in practice you only have the training data error. It turns out that if you don't fully
minimise training data error but instead push back on the complexity, your algorithm tends to
perform better.

If you deal with data + have means of fitting your data, keep in mind overfitting is often a source
of error.

## Spam Detection ##

This is a good application of machine learning.

This is a discrimination problem, can use supervised machine learning for this.

<img src="http://codegrunt.co.uk/images/ai/5-spam-detection-1.png" />

Here we are trying to determine the function which identifies email as either spam or 'ham' (legit
email). Most systems use human input, flagging spam (unflagged emails are considered ham).

Machine learning algorithm with input is email, output is spam?

### Bag of Words ###

How do we represent emails? Needs to take into account varying email contents e.g. images,
encodings, cases, etc.

Often the method used is 'bag of words' where we essentially keep a dictionary of words with word
counts, e.g.:-

    Hello I will say hello

Would become:-

    hello:2, i:1, will:1, say:1

The count is oblivious to the order of words.

We could in theory use a dictionary which doesn't actually include all the words, e.g.:-

    hello: 2, good-bye: 0

However we tend to include all words.

## Question ##

What is the probability of mail falling into spam bucket given all messages chosen at random?

3 spam messages, 5 ham messages, so 3/8 messages are spam = 3/8 probability.

## Relationship to Bayes Networks ##

### Maximum Likelihood ###

Looking at the previous example, more formally. We have the following emails:-

    SSSHHHHH

We want to determine:-

    [; p(S) = \pi ;]

Now:-

    [; P(y_i)=
       \left\{\begin{matrix}
       \pi $ if $ y_i = S\\
       1-\pi $ if $ y_i = H
       \end{matrix}\right. ;]

If instead of denoting spam 'S' and ham 'H' we denote spam 1 and ham 0:-

    11100000

    [; p(y_i) = \pi^{y_i} . (1-\pi)^{1-y_i} ;]

Assuming independence:-

    [; p(data) = \prod_{i=1}^8 p(y_i) = \pi^{count(y_i=1)}.(1-\pi)^{count(y_i=0)} ;]
    [;         = \pi^3.(1-\pi)^5 ;]

We can maximise the log of this, e.g.:-

    [; \log p(data) = 3.\log\pi + 5\log(1-\pi) ;]

The maximum of this is obtained when the derivative is 0, so let's perform the calculation:-

    [; \frac{\mathrm{d\log p(data)} }{\mathrm{d} \pi}=0 ;]
    [; = \frac{3}{\pi} - \frac{5}{1-\pi} ;]

So:-

    [; \frac{3}{\pi} = \frac{5}{1-\pi} ;]
    [; 3(1-\pi) = 5\pi ;]
    [; \pi = \frac{8}{3} ;]

Which is what we obtained previously.

So counting spam emails and dividing by total was mathematically legitimate.

## Relationship to Bayes Network ##

We're building up a Bayes network here, with parameters estimated via supervised learning via
maximum likelihood estimator.

<img src="http://codegrunt.co.uk/images/ai/5-relationship-to-bayes-network-1.png" />

At root it has an unobservable variable called 'spam' which is binary, and has as many children as
there are words in a message, where each word has an identical conditional distribution of the word
occurrence given the class spam or not spam.

In our previous example we have the following words:-

    offer
    is
    secret
    click
    sports

And we determined:-

    [; p("secret"|SPAM) = \frac{1}{3} ;]
    [; p("secret"|HAM) = \frac{1}{15} ;]

How many parameters required for a 12 word network?

Rather than the anticipated 24 + 1 (given our previous parameter count approach), the answer is
actually 22 + 1, since we know the sum of all word probabilities given S is 1, thus can obtain the
last parameter by 1 - (sum of all other probabilities given spam). The same goes for ham, so we can
subtract 2 here.

## Question 1 ##

Message m = "sports"

We want:-

    [; P(spam|m) ;]


Given our sample set:-

    SPAM

    OFFER IS SECRET
    CLICK SECRET LINK
    SECRET SPORTS LINK

    HAM

    PLAY SPORTS TODAY
    WENT PLAY SPORTS
    SECRET SPORTS EVENT
    SPORTS IS TODAY
    SPORTS COSTS MONEY

We can see from this:-

    [; P(m|spam) = \frac{1}{9} ;]
    [; P(m) = \frac{6}{24} = frac{1}{4} ;]
    [; P(spam) = \frac{3}{8} ;]

So by Bayes' rule:-

    [; P(spam|m) = \frac{\frac{1}{9} \times \frac{3}{8}}{\frac{1}{4}} = \frac{1}{6} ;]


## Question 2 ##

Message m = "secret is secret"

We want:-

    [; P(spam|m) ;]

So, via Bayes we have:-

    [; P(spam|m) = \frac{P(m|spam)P(spam)}{P(m)} ;]
    [; = \frac{P(m|spam)P(spam)}{P(m|spam)P(spam) + P(m|\lnot spam)P(\lnot spam)} ;]

We can simply count the items in each collection to determine that:-

    [; P(spam) = \frac{3}{8} ;]
    [; P(\lnot spam) = \frac{5}{8} ;]

If we assume that probability of words is conditionally independent on spam/not spam and given that
we are representing emails by bag of words, i.e. we don't care about the order, then:-

    [; P(m|spam) = P(secret is secret|spam) = P(secret|spam) \times P(secret|spam) \times P(is|spam) ;]

And similar for the ham case.

So we have:-

    [; P(m|spam) = \frac{1}{3} \times \frac{1}{3} \times \frac{1}{9} = \frac{1}{81} ;]
    [; P(m|\lnot spam) = \frac{1}{15} \times \frac{1}{15} \times \frac{1}{15} = \frac{1}{3375};]

And:-

    [; P(spam|m) = \frac{P(m|spam)P(spam)}{P(m|spam)P(spam) + P(m|\lnot spam)P(\lnot spam)} ;]

So:-

    [; P(spam|m) = \frac{\frac{1}{81} \times \frac{3}{8}}{\frac{1}{81} \times \frac{3}{8} + \frac{1}{3375} \times \frac{5}{8}} \simeq 0.9615 ;]

## Question 3 ##

M = "today is secret"

We can see straight away that 'today' isn't in the spam set, thus:-

    [; P(today|spam) = 0 ;]

And:-

    [; P(m|spam) = 0 ;]

So:-

    [; P(spam|m) = 0 ;]

## Answer and Laplace Smoothing ##

According to our model 'today is secret' is simply impossible to be spam. This isn't quite what
you'd expect. This is just overfitting. A single word shouldn't determine the outcome of our entire
model.

### LaPlace Smoothing ###

In maximum likelihood calculations, we have:-

    [; p(x) = \frac{count(x)}{N} ;]

In LaPlace smooth we use the following equation:-

    [; p(x) = \frac{count(x) + k}{N + k|x|} ;]

Where:-

* count(x) - number of occurrences of this value of the variable x.
* |x| - number of values that x can take on.
* k - smoothing parameter.
* N - total number of occurrences of x (the variable rather than value) in the sample size.

Various cases:-

1 message, 1 spam:-

    [; p(x) = \frac{1 + 1}{1 + 1 \times 2} = \frac{2}{3} ;]

10 messages, 6 spam:-

    [; p(x) = \frac{6 + 1}{10 + 1 \times 2} = \frac{7}{12} ;]

100 messages, 60 spam:-

    [; p(x) = \frac{60 + 1}{100 + 1 \times 2} = \frac{61}{102} ;]

## Question 1 ##

Assuming vocab. size = 12 and k = 1.

Since:-

    [; p(x) = \frac{count(x) + k}{N + k|x|} ;]

We have:-

    [; P(spam) = \frac{3 + 1}{8 + 1 \times 2} = \frac{2}{5} ;]
    [; P(ham) = P(\lnot spam) = 1 - P(spam) = \frac{3}{5} ;]

Assuming we keep a dictionary of 12 words:-

    [; P("today"|spam) = \frac{0 + 1}{9 + 1 \times 12} = \frac{1}{21} ;]
    [; P("today"|ham) = \frac{2 + 1}{15 + 1 \times 12} = \frac{3}{27} = \frac{1}{9} ;]

## Question 2 ##

M = "today is secret"
k = 1

We want:-

    [; P(spam|m) ;]

From our previous calculations:-

    [; P(spam|m) = \frac{P(m|spam)P(spam)}{P(m|spam)P(spam) + P(m|\lnot spam)P(\lnot spam)} ;]
    [; P(m|spam) = P("today"|spam)P("is"|spam)P("secret"|spam) ;]
    [; P(m|\lnot spam) = P("today"|\lnot spam)P("is"|\lnot spam)P("secret"|\lnot spam) ;]

We can specify the following for the |spam and |ham (i.e. [; \lnot spam ;]) instances:-

    [; P(x=word|spam) = \frac{count(word) + 1}{9 + 1 \times 12} = \frac{count(word) + 1}{21} ;]
    [; P(x=word|\lnot spam) = \frac{count(word) + 1}{15 + 1 \times 12} = \frac{count(word) + 1}{27} ;]

So:-

    [; P("today"|spam) = \frac{1}{21} ;]
    [; P("is"|spam) = \frac{2}{21} ;]
    [; P("secret"|spam) = \frac{4}{21} ;]
    [; P("today"|\lnot spam) = \frac{3}{27} = \frac{1}{9};]
    [; P("is"|\lnot spam) = \frac{2}{27} = ;]
    [; P("secret"|\lnot spam) = \frac{2}{27} ;]

And:-

    [; P(m|spam) = \frac{1}{21} \times \frac{2}{21} \times \frac{4}{21} \simeq 0.000864 ;]
    [; P(m|\lnot spam) = \frac{1}{9} \times \frac{2}{27} \times \frac{2}{27} \simeq 0.000610 ;]

We know from previous calculations that:-

    [; P(spam) = \frac{2}{5} ;]
    [; P(\lnot spam) = \frac{3}{5} ;]

So:-

    [; P(spam|m) = \frac{0.000864 \times \frac{2}{5}}{0.000864 \times \frac{2}{5}  + 0.000610 \times \frac{3}{5}} \simeq 0.486 ;]

## Summary Naive Bayes ##

So we've learnt about 'naive Bayes', where we've used maximum likelihood and a Laplacian smoother to
determine class probabilities via Bayes rule:-

<img src="http://codegrunt.co.uk/images/ai/5-summary-naive-bayes-1.png" />

This is called a 'generative model' in the conditional probabilities all aim to maximise the
predictability of individual features as if they describe the physical world.

We also use a 'bag of words' model which counts words but doesn't take into account order.

Spammers have found ways of circumventing the naive Bayes method so you have to work harder to
counter 'em.

## Advanced SPAM Filtering ##

Features you may consider:-

* From known spamming IP?
* Have you emailed this person before?
* Have 1,000 other people recently received the same message?
* Email header consistent?
* Email all caps?
* Do inline URLs point to where they say they do?
* Are you addressed by name?

Can add them to the naive base model.

## Digit Recognition ##

Naive Bayes can also be used for handwriting recognition, specifically digits.

We could do the following:-

Input vector = pixel values
Say, 16 x 16

Not vary 'shift invariant', e.g. can't recognise that the right figure below is just the left figure
shifted to the right:-

<img src="http://codegrunt.co.uk/images/ai/5-digit-recognition-1.png" />

There's many different solutions, one of which is to use smoothing, though in a different way than
discussed before. Instead of counting 1 pixel's value count, we can mix it with counts of the
neighbouring pixel values too, so we get similar statistics for a pixel whether shifted or not.

This is called 'input smoothing', technically we 'convolve' the pixels with the Guassian variable. This might give us better results.

Naive Bayes is not a good choice here, however, since the conditional independence of each pixel is too strong an assumption to make. However, it's still fun to talk about it :-)

## Overfitting Prevention ##

We've talked about Occam's razor previously - suggests a trade-off between how well we can fit the data and how 'smooth' our learning algorithm is. We've already seen LaPlacian smoothing as well as input smoothing.

The question is - how do we choose the smoothing parameter?! There is a method called 'cross
validation'. The method assumes that you have a lot of training data. However this is usually not a
problem for spam.

We begin by dividing our training data into 3 parts:-

* Train - usually around 80% of the data.
* Cross-Validation (CV) - ~ 10%.
* Test - ~ 10%.

You use the train to find all parameters, e.g. probabilities of Bayes network. You then use the cv
set to find an optimal k - you train for different values of k, observe how well the train model
performs on the cv data, leaving the test data then you maximise over all k's to get the best
performance on the cv dataset. You train agan, then test only once against the test data to verify
performance, which is the performance you report.

It's really important to split apart the cv set which is different from the test set, as otherwise
the test set would become part of the training routine, and you might overfit your training data
without realising. By keeping it separate, you get a fair answer to the question - 'how well will
your data work on future data'.

Pretty much everybody does it this way. You can redo the split between train and cv - people often
use the term '10-fold cross-validation' where they do 10 different foldings then run the model 10
times to find the optimal k or smoothing parameter.

## Classification vs. Regression ##

So far, we've been talking about supervised learning via *classification*, where the target
labels/class is discrete (in our case binary).

Often we want to predict a continuous quantity, e.g. [; y_i in [0, 1] ;]

Doing this is called 'regression'.

A regression problem could e.g. be to predict the weather/temperature which clearly involves a
continuous variable(s).

## Linear Regression ##

This isn't a binary thing anymore. More a relationship between two variables, e.g. one you know and
one you care about. You're trying to fit a curve which best represents the data.

You might try to fit a very non-linear curve, but again Occam's razor plays its part and you risk
overfitting should you get too carried away.

Data will be comprised of input vectors of length n which map to a continuous value, e.g.:-

    [; \begin{bmatrix}
        x_{11} & x_{12} & x_{13} & ... & x_{1n} & \to & y_1 \\
        x_{21} & x_{22} & x_{23} & ... & x_{2n} & \to & y_2 \\
        ...    & ...    & ...    & ... & ...    & ... & ... \\
        x_{m1} & x_{m2} & x_{m3} & ... & x_{mn} & \to & y_m \\
       \end{bmatrix} ;]

The difference between this + the classification case is that the y's are now continuous.

Once again we're looking for:-

    [; y = f(x) ;]

In linear regression we're looking at:-

    [; f(x) = w_1x + w_0 ;]

We could also do:-

    [; f(x) = wx + w_0 ;]

Where w and x are both vectors and we're calculating their inner product. However for now let's just
consider the 1-dimensional case.

## More Linear Regression ##

What are we trying to minimise?

We're trying to minimise the 'loss function'. The loss function gives us the amount of 'residual
error' after trying to fit the data as well as we possibly can.

The residual error is the sum of all training examples, minus our prediction, squared:-

    [; LOSS = \sum_j(y_j - w_1x_j - w_0)^2 ;]

This is the quadratic error between target labels and what our best hypothesis can produce.

We can write the minimisation problem as follows:-

    [; w_1^*=argmin_w L ;]

i.e. the arg min of the los over all possible vectors W.

## Quadratic Loss ##

The problem of minimising quadratic loss can be solved in closed form.

Let's look at the one-dimensional case:-

    [; L = min_w \sum(y_i - w_1x_i-w_0)^2 ;]

Minimum of this is obtained by setting the derivative of this to 0.

Partial derivation:-

    [; \frac{\partial L}{\partial w_0} = 0 = -2\sum(y_i - w_1x_1-w_0) ;]

Thus:-

    [; \sum y_i - w_1 \sum x_i = Mw_0 ;]

Hence:-

    [; w_0 = \frac{1}{M} \sum y_i - \frac{w_1}{M}\sum x_i ;]

Now if we look at [; w_1 ;]:-

    [; \frac{\partial L}{\partial w_1} = -2 \sum(y_i - w_1x_i - w_0)x_i = 0 ;]
    [; \sum_i x_i y_i - w_0 \sum x_i = w_1 \sum x_i^2 ;]
    [; \sum x_i y_i - \frac{1}{M} \sum y_i \sum x_i - \frac{w_1}{M}(\sum x_i)^2 = w_1 \sum x_i^2 ;]

Skipping some derivation:-

    [; w_1 = \frac{M\sum x_i y_i - \sum x_i \sum y_i}{M \sum x_i^2 - (\sum x_i)^2} ;]

## Problems with Linear Regression ##

Linear regression works well if the data is approximately linear.

Examples of bad situations to use linear regression:-

* Non-linear data - clearly a linear curve is not going to fit non-linear data.
* Outliers - Since we're minimising quadratic error, outliers have a hefty impact on our modelled
  curve. A very bad match is one where you

<img src="http://codegrunt.co.uk/images/ai/5-problems-with-linear-regression-1.png" />

### Logistic Regression ##

* As x tends to infinity, so does y. This is not always applicable. E.g. with weather, it's not
  correct to assume that the weather will simply become hotter or cooler. A means of dealing with
  this is to use 'logistic regression'.

There is a little more complexity here, represented by the following function:-

Given f(x), our linear function, then the output of logistic regression is obtained from:-

    [; z = \frac{1}{1 + e^{-f(x)}} ;]

This function returns a range of values from 0 - 1.

## Linear Regression and Complexity Control ##

Another problem is related to 'regularisation' or 'complexity control'. Sometimes we want a less complex model.

We achieve this via:-

    LOSS = LOSS(data) + LOSS(parameters)

    [; \sum_j(y_j - w_1x_j - w_0)^2 + \sum_i |w_i|^p ;]

Where the second function penalises the parameters.

Our regularisation term might be 'pulling' terms towards zero, along the circle if you use quadratic
error. This is done in a diamond shaped way for 'L1 regularisation'.

L1 regularisation has the advantage that it tends to make parameters 'sparse', e.g. you can drive
one of the parameters to zero. In the L2 case, parameters tend not to be as sparse, so often L1 is
preferred. E.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-linear-regression-and-complexity-control-1.png" />

## Minimising Complicated Loss Functions ##

How do we minimise more complicated cost functions? We generally have to resort to iterative methods
to minimise more complicated cost functions.

### Gradient Descent ###

We start with an initial guess, e.g. [; w_0 ;], then we update according to:-

    [; w^{i+1} \leftarrow w^i - \alpha \triangledown_w L(w^i) ;]

## Question 1 ##

Nothing to comment on.

## Question 2 ##

Nothing to comment on.

## Answer ##

Global minimum can be reached, but have to be careful to make sure the learning rate becomes smaller
and smaller over time, otherwise the prediction might bounce between values around the global
minimum and never reach it properly.

Can end up at local minimums.

Gradient descent is fairly universally applicable to more complicated problems, but you have to
check for local minima. Many optimisation books will describe tricks as to how to achieve this, we
won't go into it here.

## Gradient Descent Implementation ##

How do we minimise our cost function using gradient descent?

In our linear regression problem we use the following function (which we know we have a closed-form solution to):-

    [; L = \sum_j(y_j - w_1x_j - w_0)^2 ;]

    [; \frac{\partial L}{\partial w_1} = -2 \sum_j(y_j - w_1 x_j - w_0)x_j ;]
    [; \frac{\partial L}{\partial w_0} = -2 \sum_j(y_j - w_1 x_1 - w_0) ;]

Where we iteratively obtain our w values from:-

    [; w_1^m \leftarrow w_1^{m-1} - \alpha \frac{\partial L}{\partial w_1}(w_1^{m-1}) ;]
    [; w_0^m \leftarrow w_0^{m-1} - \alpha \frac{\partial L}{\partial w_0}(w_0^{m-1}) ;]    

## Perceptron ##

There are different ways to apply linear functions to machine learning. Not only used for
regression, but also for classification. Particularly used by an algorithm called the 'perceptron'
algorithm.

Very early model of a neuron, invented in the 1940's.

Say we had positive and negative samples, then a 'linear separator' is a linear equation which
separates the two. Clearly not all datasets have such a separator, however some do. E.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-perceptron-1.png" />

Let's start with our linear equation:-

    [; w_1 x + w_0 ;]

If higher dimensional, then [; w_1 ;] and x might be vectors. Anyway, never mind! ;-)

We can obtain class labels as follows:-

    [; 1 $ if $ w_1x + w_0 \geq 0 ;]
    [; 0 $ if $ w_1x + w_0 < 0 ;]

Which gives us our linear separation classification equation:-

    [; f(x)=\left\{\begin{matrix}
    1 $ if $ w_1x + w_0 \geq 0 \\ 
    0 $ if $ w_1x + w_0 < 0
    \end{matrix}\right. ;]

Perceptron only converges if the data is linearly separable, and if so then in converges to a linear
separation of the data, which is quite amazing :-)

Perceptron is iterative + similar to gradient descent.

### Perceptron Update ###

Start with random guess for:-

    [; w1, w0 ;]

Then update via the following:-

    [; w_i^m \leftarrow w_i^{m-1} + \alpha (y_j - f(x_j)) ;]

This is an 'online' learning rule, i.e. we don't need to process in batch, process one bit of data
at a time. Might need to go through the data many times (the j in the second part of the equation is
not an error).

This method gives us the ability to adapt our weights in proportion to the error.

If we are dead-on, i.e. our function is entirely accurate, then we make no adjustment to w, however
if not, then we make a minor correction according to a small learning rate, [; \alpha ;].

Perceptron converges to a correct linear separator if one exists.

The case of linear separation has been receiving a lot of attention in machine learning recently.

You can make different choices of linear separator, not always totally clear-cut, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-perceptron-2.png" />

Intuitively, b is a good choice, because a and c are too close to the examples such that new
examples might end up crossing the line, whereas b does not stray too close.

The area between the closest training data and the linear separator.

The margin is very important - there are is an entire class of algorithms which try to maximise the
margins, e.g.:-

* Support vector machines / SVMs
* Boosting

Very frequently used in real classification machine learning tasks.

A support vector machine derives a linear separator and actually maximises the margin. It picks a
linear separator which acts to maximise this margin. Has a robustness advantage over a typical
perceptron.

You can solve for this by solving a quadratic problem.

One of the nice aspects of SVMs is that they use linear techniques to solve non-linear
problems. E.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-perceptron-3.png" />

We use a 'kernel trick' to solve this, as clearly a linear solution won't separate positive and
negative values as we would like.

We define a third term, [; x_3 ;]:-

    [; x_3 = \sqrt{(x_1)^2 + (x_2)^2} ;]

i.e. distance from origin. This trick can help you apply non-linear techniques to a linear system,
extremely useful and quite a deep insight to see that this can help. Numerous papers out in the
literature.

Mathematically done using a 'kernel' - out of scope of class, but very powerful and useful.

### Linear Methods ###

* Regression vs. Classification
* Exact Solutions vs. Iterative Solutions
* Smoothing
* Non-Linear Problems

## k Nearest Neighbours ##

The k-nearest-neighbours method is a non-parametric machine learning method.

So far we've been talking about parametric approaches.

Parametric methods have parameters, and the number of parameters is constant + independent of
training set size (for any fixed dictionary if applicable).

Non-parametric methods - number of parameters can grow (in fact, can grow a lot over time).

## kNN Definition ##

K-Nearest-Neighbours - very simple algorithm:-

* Learning Step: memorise all data.
* Label new example: If a new example comes along whose input value you know, which you wish to
  classify, find k nearest neighbours + return the *majority* class label as your final class label
  for the example.

E.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-knn-definition-1.png" />

Here we label according to different k:-

* k=1 +
* k=3 +
* k=5 -
* k=7 -
* k=9 +

## k as Smoothing Parameter ##

As with LaPlacian smoothing before, k is a smoothing function which makes the data less scattered.

The 'Voronoi graph' shows the boundaries between positive and negative labelling by the kNN method.

k is a regulariser - larger k smoothes the output, but again there is a trade-off as things can get
misclassified. Trade-off between complexity and goodness of fit.

## Problems with kNN ##

Problems are:-

* Very large data sets

Lengthily searches for k nearest neighbours. Can be mitigated somewhat, via 'kdd trees', etc. -
makes search logarithmic.

* Very large feature spaces

Computing kNN, as the feature space of the input vectors increases becomes increasingly difficult +
the tree structure becomes more brittle.

The more features, the easier it is to select something far away, and soon it gets out of control
(e.g. exponential).

kNN works well for perhaps 3-4 dimensions, more is dodgy.

## Congratulations ##

Learnt a lot, focused on supervised machine learning - input vectors and given output labels, and
the goal is to predict the output labels given the input vectors.

Also looked at parametric models like naive Bayes, or non-parametric models like kNN. Looked at
classification vs. regression.

One of the more interesting parts of AI, dealing with big datasets. Only becoming more important.
