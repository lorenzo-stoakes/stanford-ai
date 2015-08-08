AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 9 - Planning under Uncertainty
-----------------------------------

## Introduction ##

Talking about planning under uncertainty - really brings together previous material.

## Planning Under Uncertainty MDP ##

We've talked about planning, uncertainty, and learning separately.

Now we combine them - the overlap between planning and uncertainty can be tackled with Markov
Decision Processes, or MDPs, and Partially-Observable Markov Decision Processes, or POMDPs, which
we're tackling in the upcoming classes.

Later we have a class coming up which combines all 3 - planning, uncertainty and learning, via
Reinforcement Learning (RL).

    +------------------------------+------------------------------+------------------------------+
    |                              |Deterministic                 |   Stochastic                 |
    +------------------------------+------------------------------+------------------------------+
    |Fully Observable              |A*, Depth First, Breadth First|MDP                           |
    +------------------------------+------------------------------+------------------------------+
    |Partially Observable          |???                           |POMDP                         |
    +------------------------------+------------------------------+------------------------------+

How do we represent a Markov Decision Process? One means is via a graph:-

<img src="https://ljs.io/img/ai/9-planning-under-uncertainty-mdp-1.png" />

A finite state machine becomes Markov when the outcomes of actions are somewhat random.

A Markov model consists of:-

    states  [; s_1 ... s_n ;]
    actions [; a_1 ... a_k ;]
    state transition matrix
    [; T(s, a, s') = P(s'|a, s) ;]
    Reward function
    [; R(s) ;]

For the sake of the lectures we attach rewards to states which act as the goals, such that we can
find actions which maximise total reward.

## Robot Tour Guide Examples ##

See the videos :-)

## MDP Grid World ##

Simple example:-

<img src="https://ljs.io/img/ai/9-mdp-grid-world-1.png" />

Clearly, we can't just use a conventional planning approach, as we have stochasticity in the outcome
of actions. We want a policy, which assigns actions to *any* state, e.g.:-

    [; \pi(S) \rightarrow A ;]

For each state other than the absorbing states we have to define an action in order to define a
policy. The planning problem becomes one of finding the optimal policy.

## Problems with Conventional Planning 1 ##

In order to understand the usefulness of a policy, let's consider the application of conventional
planning to the same problem:-

<img src="https://ljs.io/img/ai/9-problems-with-conventional-planning-1.png" />

Here we're branching off a *lot* because of all the possible outcomes, so a serious problem with
using a conventional approach is a **large branching factor**.

## Branching Factor Question ##

Consider how many other states you can reach from a given state (this is the *branching factor*),
e.g. from c1 you can reach 3 - what is the branching factor for b3?

(assuming you can possibly go to the neighbours of any square you are attempting to get into)

The answer is 8 - b3, a2, a3, a4, b4, c4, c3, c2.

## Problems with Conventional Planning 2 ##

There are other problems with the search approach.

We've already mentioned a large branching factor, however there are others:-

* Branching factor large - as discussed previously,
* Tree too deep - we might well go on 'circling' states, i.e. considering states for a very long
  time, possibly even encountering infinite loops.
* Many states visited more than once - given the stochastic elements of the state space, especially
  given it is possible to not move at all, it is easy to visit a state more than once using search.

These are all overcome by our policy method. It is far better to use policies in stochastic
environments.

## Policy Question 1-4 ##

Note that for each direction you attempt to move in there's an 80% chance you'll move in that
direction and a 10% chance you will move to the left of that direction, and 10% you'll move to the
right, e.g. trying to move north - there's an 80% chance you'll move north, a 10% chance you'll move
west and a 10% chance you'll move east, e.g:-

<img src="https://ljs.io/img/ai/9-policy-questions-1-4-1.png" />

Sometimes choices in a policy are counterintuitive - you want to avoid a certain action, so you
choose to move in a direction which is into a wall since there's an 80% chance you'll get nowhere,
but a 20% chance you'll move to a safe square.

Good policy choices:-

* a1 - move E
* c1 - move N
* c4 - move S
* b3 - move W

## MDP and Costs ##

Even though we've got no cost associated with movement, other than the 2 absorption costs, it turns
out that even this simple scenario is non-trivial.

Essentially the example is frustrating because you are choosing to run into the wall given certain
scenarios. A lot of this is due to the fact we've associated no cost to movement, whereas in the
real world, movement is not free.

Let's look at a reward function which takes movement cost into account:-

    [; R(s) \rightarrow \left\{\begin{matrix}
       +100 & b4\\
       -100  & c4\\
       -3 & other\ states
       \end{matrix}\right. ;]

An MDP needs to maximise rewards at all times, e.g.:-

    [; E[\sum_{t=0}^\infty R_t] \rightarrow max ;]

I.e. we seek to maximise the *expectation* of rewards over time.

We can also introduce a *discount factor*, [; \gamma ;]:-

    [; E[\sum_{t=0}^\infty \gamma^t R_t] \rightarrow max ;]

It decays future reward relative to immediate award. Effectively an alternative means of specifying
cost. It's also useful because it forces us to get to the goal as quickly as possible.

Mathematically, it keeps the expectation bounded. We can show that the expectation will always be:-

    [; \leq \frac{1}{1-\gamma} | R_{max} | ;]

## Value Iteration 1-3 ##

We define a value for each state:-

    [; V^\pi = E_\pi[\sum_t \gamma^t R_t | s_0 = s] ;]

This is essentially telling us the average reward before hitting a goal state for a given policy
[; \pi ;].

By calculating value functions, we will find better + better policies.

Value iteration recursively calculates a value function so we obtain the 'optimal value function' so
we can derive the optimal policy.

Let's examine a case where all the state values are initially zero apart from the absorbing states:-

<img src="https://ljs.io/img/ai/9-value-iteration-1-3-1.png" />

Imagine we want to determine a good value for a3. We work backwards from a4 - knowing there's an
80% probability of moving east and getting the +100, and there's a cost of 3 associated with the
state, we can calculate an appropriate value as follows:-

    [; V(a3, E) = 0.8 \times 100 - 3 = 77 ;]

After applying this iteratively throughout all states in the grid we get the following values:-

<img src="https://ljs.io/img/ai/9-value-iteration-1-3-2.png" />

Let's define a formula for value iteration:-

    [; V(s) \leftarrow [ max_a \gamma \sum_{s'}P(s'|s, a)V(s')] + R(s) ;]

This looks complicated but really is not - we're maximising over all possible actions (the max term
here), and simply summing up the probabilities of all successor states given a given action and
initial state multiplied by its value.

The equation is referred to as 'back-up', and if we take into account terminal states we have:-

    [; V(s) \leftarrow \left\{\begin{matrix}
       R(s) $ if s is terminal$\\
       [ max_a \gamma \sum_{s'}P(s'|s, a)V(s')] + R(s)
       \end{matrix}\right. ;]

This process of updates is guaranteed to *converge*.

Once we have converged the arrow turns into an equals sign and we have a Bellman equality or Bellman equation.

## Deterministic Question 1-3 ##

Consider our grid, only now things are deterministic.

Assume:-

    [; \gamma = 1 ;]
    [; cost = -3 ;]

E.g.:-

<img src="https://ljs.io/img/ai/9-deterministic-question-1-3-1.png" />

    V(a3) = 97

Since there is a clear maximising action here, i.e. moving east, and given it's deterministic, our
value is simply 100 - 3 = 97.

    V(b3) = 94

Moving north is the only reasonable move here given the deterministic situation.

Similarly:-

    V(c1) = 85

Note these are the *first* values rather than the final ones (though in this case, the determinism
would mean you don't have to worry about the values being different).

## Stochastic Question 1-2 ##

In the stochastic case, since initial values for staying in place or moving south are 0:-

    [; V(a3) = 100 \times 0.8 + 0.1 \times 0 + 0.1 \times 0 + -3 = 77 ;]

If we assume we ought to go west we get:-

    [; V(b3) = 0 \times 0.8 + 77 \times 0.1 + 0 \times 0.1 + -3 = 4.7 ;]

If we assume we ought to go north we get:-

    [; V(b3) = 77 \times 0.8 + -100 \times 0.1 + 0 \times 0.1 + -3 = 48.6 ;]

Which is the correct value (for this iteration!). We discount moving south as that is clearly all
downside.

## Value Iterations and Policy 1-2 ##

If we look at our backup function:-

    [; V(s) \leftarrow [ max_a \gamma \sum_{s'}P(s'|s, a)V(s')] + R(s) ;]

Then it is clear which action we need to take in a given policy, i.e. the one which maximises the
value.

We can define our optimal policy via:-

    [; \pi(s) = argmax_a \sum_{s'}P(s'|s, a)V(s') ;]

We simply choose the best thing to do given the already calculated backed-up values.

If we look at the situation where we have [; \gamma = 1 ;] and R = -3, then we have the following
values and associated optimal policy:-

<img src = "https://ljs.io/img/ai/9-value-iterations-and-policy-1-2-1.png" />

This is very different if we have cost 0:-

<img src = "https://ljs.io/img/ai/9-value-iterations-and-policy-1-2-2.png" />

Even though it's odd that all states have value 100, this is as expected, since we can guarantee we
are always going to get to the 100 absorbing state, and moving has no cost.

If we consider the case where R=-200, i.e. each non-absorbing state has a cost of 200, then the
policy becomes one of getting out of the game as quickly as possible to avoid paying the cost of 200
any longer:-

<img src = "https://ljs.io/img/ai/9-value-iterations-and-policy-1-2-3.png" />

## MDP Conclusion  ##

We've learned about Markov decision processes which are

* Fully observable with states:-
    [; s_1 ... s_n ;]
And actions:-
    [; a_1 ... a_k ;]
* Stochastic, i.e. we consider:-
    [; P(s'|a, s) ;]
* Rewarded per state by:-
    [; R(s) ;]
* Has objective:-
    [; E \sum_t \gamma^t R^t \rightarrow max ;]
* We used *value iteration* to solve problems, using values for given states via:-
    [; V(s) ;]
Noting that there is an alternative means of assigning value to an approach by looking at state,
action pairs via:-
    [; Q(s, a) ;]
Though we haven't considered this approach yet.
* We note that this converges via:-
    [; \pi(s) = argmax_a \sum_{s'}P(s'|s, a)V(s') ;]

## Partial Observability Introduction ##

Let's examine the partially observable case - we don't want to go into too much depth here however,
given the complexity. However, want to give a good flavour of it. Also the case in the actual
Stanford class :-)

## POMDP vs MDP ##

POMDP addresses problems of optimal exploration vs. exploitation, where some of the actions might be
information-driven actions and others goal-driven. Not applicable to MDPs as in that case the state
space is fully observable, therefore no need for information gathering.

## POMDP ##

Let's consider a very simple environment, a maze:-

Let's consider a deterministic fully observable case:-

<img src="https://ljs.io/img/ai/9-pomdp-1.png" />

Now a stochastic fully observable case:-

<img src="https://ljs.io/img/ai/9-pomdp-2.png" />

Now let's consider a partially observable case:-

<img src="https://ljs.io/img/ai/9-pomdp-3.png" />

Assuming that there is a sign which tells the agent where to go.

Here, if we knew which of the two exits was +100, we'd go north straight away, however we instead
ought to go south purely for the purposes of information gathering. The question is, can we plan
such that we can take this into account?

You can't just, e.g. average the two possible cases (two 'worlds', one where the left exist is the
+100 exit, and the other where the right exist is the +100 exit), e.g.:-

<img src="https://ljs.io/img/ai/9-pomdp-4.png" />

As this will always cause the agent to go north, and it makes impossible for the agent to go south
which is clearly the correct thing to do here. Also when the agent arrives at the intersection, it
doesn't know what to do.

What does work is considering an information space/belief space on which you can superimpose the MDP
approach of value iteration, e.g.:-

<img src="https://ljs.io/img/ai/9-pomdp-5.png" />

## Planning Under Uncertainty Conclusion ##
 
We've talked about MDPs, POMDPs, information spaces, and now it's even possible to apply it.
