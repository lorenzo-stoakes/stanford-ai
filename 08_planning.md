AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 8 - Planning
-----------------

## Introduction ##

We define AI as the study and process of finding appropriate actions of an agent. In a sense,
planning is the core of AI. We've looked at problem-solving search over a state space using
e.g. A\*. Given a state space we can find a path to the goal. These approaches are great for
deterministic, fully-observable environments. We'll see how to relax those constraints in this unit.

## Problem Solving vs. Planning ##

Looking back to problem solving so far - we have a start point, and a goal, the objective is to find
a path between the two. Imagine if you were to do that in the real world, i.e. you did all the
planning up front, then simply executed the path without reacting to the environment at all.

Some amusing diagrams of experiments where people have been told to walk in a straight line but
clearly simply can't! There is a good example, where a hiker decided to look at shadows, and by
keeping them all going in a single direction, was able to walk straight.

The moral is - need feedback from the environment. Have to interleave planning and executing, can't
just do everything upfront.

## Planning vs. Execution ##

Properties of the environment make massive upfront planning a problem:-

* Stochastic - the property that makes it most inappropriate is if the environment is stochastic,
i.e. we don't know for sure what an action is going to do. We need to deal with contingencies,
e.g. I tried to move forward and the wheels slip/brakes not effective/people's legs don't go
entirely forward/traffic lights - the results of going through the traffic lights when the light is
red is likely to be different than if it is green ;-).
* Multiagent - If there are other agents involved, then we have to plan around what they're going to
  do, and react when they do something unexpected. We can only do this at execution time, not
  planning time.
* Partial observability - Let's say we have a plan to go from A -> S -> F -> B, however at S the
  road that goes to F has a sign which might indicate that the road to F is closed. We can only know
  what is on that sign by going there. Another way to look at it is that we don't know what state we
  are when we are at A - are we in the state where we are at A but the road is open, or are we in
  the state where we are at A but the road is closed? We only know this once we get to S.

Other factors:-

* Unknown - We might have lack of knowledge on our part - e.g. incomplete maps.
* Hierarchical - The plan might be very high-level, however the realities of what we can actually do
  are quite low-level, e.g. path we want to follow vs. actually steering, etc.

Most of these problems can be resolved by changing our point of view - rather than viewing things in
terms of world states, we can view them in terms of 'belief states'.

## Vacuum Cleaner Example ##

Looking at the vacuum example again:-

<img src="http://codegrunt.co.uk/images/ai/8-vacuum-cleaner-example-1.png" />

If we have a fully-observable, fully-deterministic world, then it's easy to plan. Imagine if the
sensor failed, such that you no longer know where the vacuum is, or whether there is any dirt at the
vacuum's location. All you can do is essentially draw a big box around the entire state space as
shown and essentially say 'I know I'm somewhere in here'.

We believe that we are in one of the 8 states, and when we execute an action then we are going to
get to another belief state.

## Sensorless Vacuum Cleaner Problem ##

This is the belief state space for the sensorless vacuum cleaner problem:-

<img src="http://codegrunt.co.uk/images/ai/8-sensorless-vacuum-cleaner-problem-1.png" />

Here we have accepted the case that we can't know where we are, however amazingly we can actually
determine things about the environment simply by executing actions.

Note that in the real world, say moving left and right, are inverse of each other - so if you move
left then move right, you end up back in the same position. This is not the case in belief state
spaces, as if you move right then left again, you have still reduced the possible number of states
by 4.

It is possible to have plans which can reach the goal without ever observing the world. Plans like
this are known as 'conformant' plans. For example, if our objective is the clean the current
location, then all we need to do is suck.

## Partially Observable Vacuum Cleaner Example ##

We've been considering sensorless planning in a deterministic world. Let's look at
partially-observable planning (still in a deterministic world).

Imagine we have local sensing - e.g. the vacuum can determine whether there is dirt in the current
location (but it can't see if there is dirt in another location).

Let's look at a diagram:-

<img src="http://codegrunt.co.uk/images/ai/8-partially-observable-vacuum-cleaner-example-1.png" />

Once we move from one state to another, we can then observe the world and split our belief state
depending on the observance. This is part of the act-observe cycle.

In a deterministic world, each world state maps into exactly one other one (this is by definition),
so the size of the belief state will remain the same, or reduce if two actions bring you to the same
place, somehow. Observation works in the opposite way - we partition the belief state. Observations
can't bring new world states into a belief state, they can only divide what already
exists. Observations can't make things worse - they could be useless, however they're not going to
increase the size of the belief state.

## Stochastic Environment Problem ##

Let's consider a robot that has slippery wheels, i.e. sometimes it moves, and sometimes it does not,
but the suck operation always works perfectly:-

<img src="http://codegrunt.co.uk/images/ai/8-stochastic-environment-problem-1.png" />

Note that actions can result in an *increase* in the belief state, since we don't know what the
result of the action is going to be. Stochastic means that there are multiple outcomes for a given
action.

No matter what plan you make, you can never be sure the wheels haven't slipped. Perhaps with
infinite sequences? Examine next.

## Infinite Sequences ##

In this new notation, instead of writing plans in a linear sequence such as [S, R, S], let's write
them as a tree structure:-

<img src="http://codegrunt.co.uk/images/ai/8-infinite-sequences-1.png" />

Can write as:-

    [ S, while A: R, S ]

If the stochasticity is independent, i.e. sometimes it works and sometimes not, then with
probability 1 in the limit, then this will achieve the goal. We can't bound the number of steps in
which it is going to succeed, only that it is guaranteed at infinity.

## Finding a Successful Plan ##

How do you find a plan in a stochastic environment? It is similar to search in problem solving, only
a little more complicated. We have branches which are a part of the plan rather than of the search
tree itself. We find a portion of the tree which successfully reaches the goal state, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/8-finding-a-successful-plan-1.png" />

## Question ##

What can we guarantee about the plan?

### Unbounded Solution ###

What guarantees an unbounded solution?

Every leaf node needs to be a goal, because we can't control the observations, and they
might go one way or another, however we want every possible observation to result in a solution.


### Bounded Solution ###

How do guarantee a bounded solution?

We mustn't have loops, as if we do, then there is no guarantee that we will reach a goal in a finite
amount of time.


## Problem Solving via Mathematical Notation ##

We can write plans using a more formal notation, e.g. we can write:-

    [; [A, S, F] $Result$($Result$(A, A \rightarrow S), S \rightarrow F) \in $Goals$ ;]

In a fully-observational world we write:-

    [; s' = Result(s, a) ;]

However when dealing with belief states we can write:-

    [; b' = Update(Predict(b, a), o) ;]

Where a is action, b is belief state, o is observation.

This is known as the predict update cycle.

## Tracking the Predict Update Cycle ##

Here's an example of tracking the predict-update cycle:-

<img src="http://codegrunt.co.uk/images/ai/8-tracking-the-predict-update-cycle-1.png" />

This is in a situation where the actions are guaranteed to work as advertised, i.e. if you suck then
it will suck up the dirt, if you move then you actually move. However we will call this the
kindergarten world where toddlers can deposit dirt in any location at any time.

Some of the belief states get quite large. We could instead of listing through each of the possible
states simply have variables for the 3 elements involved - where is the vacuum, is the left side
dirty, is the right side dirty. We could then have some formula over these variables to describe
states.

This way, some of the large belief states can be made small in description.

## Classical Planning 1 ##

This is a notation which is a representation language for dealing with states, actions + plans, and
is also an approach for dealing with complexity by factoring the world into variables.

### State Space ###

Under classical planning, the state space consists of all the possible assignments to k-boolean
variables, i.e. [; 2^k ;] states i this state space.

In vacuum world we can have 3 variables:-

* Dirt in A
* Dirt in B
* Vacuum in A

So there are 8 possible states in this world, succinctly represented by the 3 variables.

### World State ###

A world state is the complete assignment of the k boolean variables.

### Belief State ###

Depends on what type of environment you are dealing with:-

* Complete Assignment - Complete assignment of the k boolean variables - in core classical planning,
  belief state had to be a complete assignment, which is useful for dealing with deterministic,
  fully-observable domains.
* Partial Assignment - This is where some of the k boolean variables have values, and some don't.
* Arbitrary Formula - Any formulation we want.

What do actions look like, and what do results of action look like?

These are represented in classical planning by an action schema - 'schema' because it represents
many possible similar actions.

E.g. we want to send cargo around the world.

Let's look at the action schema for having a plane fly from one location to another:-

    Action(Fly(p, x, y)
           PRECOND: Plane(p) [; \wedge ;] Airport(x) [; \ ;] Airport(y) [; \wedge ;] At(p, x)
           EFFECT: [; \lnot ;] At(p, x) [; \wedge ;] At(p, y))

This is the set of all possible actions for all x and all y. It says what we need to know in order
to apply the action (the preconditions), and what will happen, i.e. the transition of state spaces.

These look like first-order logic, however we're dealing with a propositional world, where we are
essentially concatenating variable names together to obtain a new variable, which is essentially
done to make it easier to cover all potential flying actions.

## Classical Planning 2 ##

Here is a more complete representation of a problem-solving domain in classic planning:-

<img src="http://codegrunt.co.uk/images/ai/8-classical-planning-2-1.png" />

Init is initial state, goal is the target goal.

How do we do planning using this?

## Progression Search ##

The simplest way to do planning is to do it the exact same way we do it in problem solving, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/8-progression-search-1.png" />

This is known as 'forward' or 'progression' state space search. We're searching through the space of
exact states deterministically.

This is essentially what we had before, but there are now possibilities which we didn't have before
available given the representation.

## Regression Search ##

This is known as 'backwards' or regression search.

We start with the goal state and work backwards, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/8-regression-search-1.png" />

Note that the goal state is the complete goal state. Note that the goal state here isn't incomplete,
it contains literally all we know about the state, as other variables can be whatever they want to
be, we simply don't care.

We then work backwards, thinking 'what actions would bring me to here', something we could do in
problem solving with a single goal state - we could ask 'what other arcs are coming in to that goal
state'. This goal state doesn't represent a single state, rather it represents a family values, so
we can't just look at incoming arcs. What we can do instead is look at possible actions which could
possibly result in this value.

We can have a look at what could set some cargo to be at a location, and it's clear that the
'unload' action fits this criteria. If we look at cargo [; C_1 ;] which is required at JFK, then we
have p unknown (we don't know the plane) and airport JFK by definition.

We can use an arrow to represent not only a single action, but a set of possible actions, e.g. here
any plane unloading cargo at the destination, JFK. We can regress the state over the operator, with
several unknowns (like the goal state, which has unknowns in variables unrelated to the goals).

We continue searching backwards until we reach the initial state, which is our solution which we can
then apply forwards.

## Regression vs. Progression ##

Let's look at an example where backwards search makes sense.

We define a world where we need to buy a book:-

    Action(Buy(b)
           PRE: ISBN(b)
           EFFECT: OWN(b))

    Goal(Own(0136042597))

We leave out considerations of money for simplicity.

We start in the initial state of not knowing anything. If there are, let's say, 10M ISBN numbers,
then there are 10 million branches coming out of the initial node. We'd have to try every last one
in forward search, which is clearly not especially efficient. So it is better to start at the goal,
e.g.:-

<img src="http://codegrunt.co.uk/images/ai/8-regression-vs-progression-1.png" />

## Plan Space Search ##

There is one more search we can do with classic planning which we couldn't do previously, which is
to search through the space of plans rather than states, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/8-plan-space-search-1.png" />

In forward search we were searching through concrete world states, in backwards search we were
searching through abstract states, but in plan space search we search through the space of plans.

The initial plan is clearly flawed as it doesn't do anything. So we add an operator to the plan in
each step (here concerned with somebody getting dressed).

This approach used to be popular in the '80s, however it has now faded from popularity. The most
popular approaches tend to be forward search. The advantage of forward search is that we have good
heuristics. We've seen how important it is to have good heuristics to do a good heuristic
search. And because the forward search works with concrete plan states, it tends to make choosing
good heuristics easier.

## Sliding Puzzle Example ##

To understand choose of heuristics, let's have a look at the sliding puzzle again. Here is part of
the state space for the 8-puzzle:-

<img src="http://codegrunt.co.uk/images/ai/8-sliding-puzzle-example-1.png" />

Let's try and determine what the action schema looks like:-

    Action(Slide(t, a, b)
           PRE: On(t, a) [; \wedge ;] Tile(t) [; \wedge ;] Blank(b) [; \wedge ;] Adj(a, b)
           EFFECT: On(t, b) [; \wedge ;] Blank(a) [; \wedge \lnot ;] On(t, a) [; \wedge \lnot ;] Blank(b))

A human analyst could come up with heuristics to find a better solution. We can however do this
automatically by relaxing the problems, e.g.:-

    Action(Slide(t, a, b)
           PRE: On(t, a) [; \wedge ;] Tile(t)
           EFFECT: On(t, b) [; \wedge ;] Blank(a) [; \wedge \lnot ;] On(t, a))

Here we've eliminated the requirement for b to be blank, a and b to be adjacent and b to no longer
be blank, each of which are useful heuristics.

Because we have the logic in the form of classical planning, we can have a program come up with
heuristics rather than a human.

## Situation Calculus 1 ##

This is one more representation for calculation.

Say I want to move all the cargo from A to B. You can't express the idea of 'all' in propositional
languages like classical planning, but you can in first-order logic.

There are a lot of ways to use first-order for logic, the best known being situation calculus. Not a
new kind of logic, rather it's the same old first-order logic with a convention for representing
states and actions.

Conventions are:-

* Actions:objects - Actions are represented as objects, typically functions, e.g. Fly(p, x, y)
* Situations:objects - Situations are represented as objects which correspond to paths rather than
  states, the paths of actions. If you arrived at the same world state by two separate actions,
  they'd be considered two separate situations in situation calculs. We have an initial situation
  (often called [; S_0 ;]) and we have a function on situations called Result, so
  [; S' = Result(S, a) ;].

  Instead of describing the actions that are applicable in a situation with a predicate Actions(s),
  we describe the actions that are *possible* in a state, using the Poss predicate - Poss(a, s)
  means is action a possible in state s.

  There's a specific form for describing these predicates, e.g.:-

    somePrecond(s) [; \Rightarrow ;] Poss(a, s)

The possibility axiom for the action fly is:-

    [; $Plane$ (p, s) \wedge $Airport$(x, s) \wedge $Airport$(y, s) \wedge $At$(p, x, s) \Rightarrow Poss(Fly(p, x, y), s) ;]

## Situation Calculus 2 ##

There's a convention in situation calculus that predicates such as At(p, x, s), which can vary from
one situation to another, are known as fluents (coming from fluent referring to fluidity/change over
time) and are placed as the last argument in the predicate.

The hardest part of situation calculus is describing what changes and doesn't change as a result of
an action. In action schemas, we described one action at a time and what changes. In situation
calculus it is easier to do it the other way round, i.e. instead of writing one schema for each
action, we write one for each fluent/predicate that can change.

We use a convention called 'successor state axioms'. These are used to describe what happens in the
state that's a successor of executing an action.

In general they will have the form of:-

    [; \forall a, s $ Poss$(a, s) \Rightarrow ($fluent true$ \Leftrightarrow $a made it true$ \vee $ a didn't undo it$) ;]

E.g. the successor state axiom for the in predicate (note we leave out the [; \forall ;] quantifiers):-

    [; Poss(a, s) \Rightarrow $ In$(c, p, $ Result$(s, a)) \Leftrightarrow (a=$ Load$(c, p, x) \vee ($ In$(c, p, s) \wedge a \not = $ Unload(c, p, x))) ;]

In English - for all a and s for which it's possible to execute a in situation s, the in predicate
holds iff the action was a load or the in predicate used to hold in the previous state and the
action is not an unload.

## Situation Calculus 3 ##

Let's say:-

    Initial State: [; S_0 ;]
               At([; P_1 ;], JFK, [; S_0 ;]) [; \forall c $ Cargo(c)$ \Rightarrow $ At$(c, JFK, S_0) ;]

    Goal: [; \exists s, \forall c $ Cargo(c)$ \Rightarrow $ At$(c, SFO, s) ;]

This essentially says 'move all the cargo from JFK to SFO'

We can essentially use any valid sentence from first-order logic here which gives us a lot of power.

The great thing about situation calculus is that once we have described this in situation calculus,
we don't need any special programs to find a solution, because we already have theorem provers for
first-order logic and can use them to find a path which satisfies the goal.

The advantage of situation calculus is that we have the full power of first-order logic, we can
represent anything we want, far more flexibility than problem solving or classic planning.

So overall we've seen many ways to deal with planning. We started with deterministic,
fully-observable environments, and moved to stochastic and partially-observable environments. And we
were able to distinguish between plans which can or can not solve a problem. However all of these
approaches have a weakness - we can't distinguish between probable and improbable solutions, which
will be the subject of the next unit.
