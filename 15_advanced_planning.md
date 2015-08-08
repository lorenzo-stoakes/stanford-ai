AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 15 - Advanced Planning
---------------------------

## Introduction ##

We're returning to the topic of planning. Looking at 4 topics we left out last time we looked at
it:-

* Time - Not just looking at actions which occur before or after another given action, also looking
  at actions which persist over a length of time.
* Resources necessary to do a task.
* Active perception - taking the action of perceiving something.
* Hierarchical plans - Plans consisting of steps with substeps.

## Scheduling ##

First we start with time - looking at a series of tasks with durations.

Consider a task network:-

<img src="https://ljs.io/img/ai/15-scheduling-1.png" />

Consists of start and finish node, with task nodes between and associated time requirements.

We essentially want to determine start times for each so we finish as soon as possible.

## Schedule Question ##

A schedule is defined in terms of specifying for every task in the network the earliest start time
(ES), and the latest possible start time (LS) for which it's possible to complete the total network
in the shortest possible time.

We determine these via a set of recursive formulas which can be solved via dynamic programming:-

    [; ES(s) = 0 ;]
    [; ES(B) = max_{A \rightarrow B} ES(A) + Duration(A) ;]
    [; LS(F) = ES(F) ;]
    [; LS(A) = min_{B \leftarrow A} LS(B) - Duration(A) ;]

## Resources Question ##

We want to consider resources.

We could, theoretically, describe resources purely using classical planning, e.g.:-

<img src="https://ljs.io/img/ai/15-resources-question-1.png" />

The goal is not achieved as we have only 4 nuts.

We need to 'backtrack' for each nut and bolt so a depth-first tree search needs to consider 4!5!
possible paths - all combinations of nuts and all combinations of bolts. I'm not sure I understand the reasoning behind this!

The problem is because we consider nuts and bolts uniquely whereas we should consider them to be
identical rather than distinct so we can handle them more efficiently.

## Extending Planning ##

We've extended the planning syntax to handle this:-

<img src="https://ljs.io/img/ai/15-extending-planning-1.png" />

Here we've added a new type of statement which declares resources + how many we have of each. We
also explicitly model inspectors.

Actions have two new types of clauses - the fasten action has a CONSUME clause which 'consumes'
resources and once they're used they're gone forever.

The USE clause indicates that the resource in question (here, for the action 'inspect', we use the
inspector resource) is in use while the action is being performed. Once the action has completed the
resource is returned to the pool for use later.

Doing things this way eliminates the exponential explosion unadjusted planning introduces when using
resources.

## Hierarchical Planning ##

We want to close the 'abstraction gap'

In a lifetime you live of order [; 10^9 ;] seconds, during that time you have a choice of actions to
take, with around [; 10^3 ;] muscles, and perhaps [; 10^1 ;] actions possible per second, so in a
lifetime you have around [; 10^{13} ;] actions available, give or take an order of magnitude or two.

There's a big gap between the [; 10^{13} ;] actions available here and the approximately [; 10^4 ;]
actions current planning software can handle.

Part of the problem is that it's difficult to deal with actions at the level of the movement of a
muscle - we'd rather deal with more abstract plans.

Will introduce the notion of a 'Hierarchical Task Network' (HTN) - and rather than having a planned
sequence of individual steps, can consider smaller number, but where a number of steps can
correspond to more than one other. This idea is known as 'refinement planning'.

## Refinement Planning ##

Consider the following:-

<img src="https://ljs.io/img/ai/15-refinement-planning-1.png" />

In addition to standard actions, we have abstract actions, e.g. going home to the San Francisco
Airport, etc.

Can determine a complex plan at the abstract level of e.g. 'navigate' rather than having to work out
a path from a, b to x, y.

An HTN achieves the goal if, for every part, at least one refinement achieves goal.

Is an and/or choice.

## Reachable States + Question ##

As well as doing an and/or search, we can solve an abstract a HTN planning problem without going all
the way down to the concrete steps. E.g.:-

<img src="https://ljs.io/img/ai/15-reachable-states-question-1.png" />

Here the start state is highlighted, and the goal state is in grey.

We've highlighted one abstract action and the set of states that can be reached by the abstract
action, if we refine the abstract action by using one concrete action or another.

Similar to belief states, when we were moving from one state to several other possible states. Here
it's different - rather than the uncertainty of which state we'll end up being down to
stochasticity, the uncertainty is due to the choice we make as to which refinement we're going to
use.

We can determine whether it's possible to find a refinement by seeing if the subsequent belief
states intersect with the goal states.

In order to find the correct actions to take we can work back from goal state to the start.

It can be difficult to exactly specify what states are reachable by an abstract action because the
refinements are complicated.

Can consider an *approximate* set of reachable states:-

<img src="https://ljs.io/img/ai/15-reachable-states-question-2.png" />

Here there are upper and lower bounds on states which are reachable (solid line contains lower
bound, dashed line upper bound) - consider a trip to the San Francisco airport - it will take at
least 1/2 - 1hr to get there no matter which route chosen. This is a lower bound. Other aspects
depend on choice made - might spend money, might use petrol.

## Conformant Plan Question ##

Will consider how to extend classical planning to allow active perception to deal with partial
observability.

Consider the following problem description:-

<img src="https://ljs.io/img/ai/15-conformant-plan-question-1.png" />

The active perception here is the LookAt action.

We now have percept schemas as well as action schemas. We've introduced a new variable - c in
percepts - meaning we are given new information by this action.

A conformant plan is one which does not perform sensing.

This plan is conformant as we can just go ahead and paint without knowing what colour we're using,
so the percept isn't necessary.

## Sensory Plan Question ##

A problem with this plan is that, if the chair and table were already the same colour, we could have
wasted our time in attempting to paint them unnecessarily.

In this plan we can use the following logic:-

<img src="https://ljs.io/img/ai/15-sensory-plan-question-1.png" />

Which minimises the effort required.
