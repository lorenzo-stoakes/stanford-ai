AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 10 - Reinforcement Learning
--------------------------------

## Introduction ##

Learned how MDPs can be used to decide an optimal set of actions for an agent in a stochastic
environment. However, in order to do so you need to know where the rewards and penalties are.

Reinforcement learning can tell you this.

## Successes ##

Looking at the grid again:-

<img src="http://codegrunt.co.uk/images/ai/10-successes-1.png" />

Imagine that we didn't know where the rewards were here. We can have the agent explore the
territory, find the rewards and determine an optimal policy.

Analogous to a game, like backgammon, which is stochastic (i.e. the dice rolls):-

<img src="http://codegrunt.co.uk/images/ai/10-successes-2.png" />

In the 1990s, Gary Tesauro at IBM wrote a program which played Backgammon. He started by having
expert Backgammon players label example positions, such that he could determine the utility of a
game state, u(s). However, this turned out to be tedious work for the experts, so only a few
examples were labelled, and eventually it turned out that this approach did not result in effect
play.

He then tried to tackle the problem in a different way, by having the engine play against itself,
and whenever it won that strategy was given a positive score, and vice-versa. This was an
application of reinforcement learning. The results were considerably better, and produced an engine
which could play at the level of the best players in the world. He used around 200,000 examples of
games, however that was small compared to the state space of backgammon (around a billionth).

Another example is a model helicopter which Andrew Ng trained to perform manoeuvres, and achieved
this by looking at only a few hours of expert helicopter pilots who would take over the controls and
pilot the helicopter. There'd be rewards for doing things well and punishments for doing it badly,
such that he was able to build an automated pilot from just those examples via reinforcement learning.

## Forms of Learning + Question ##

Let's examine the 3 forms of learning:-

* Supervised learning has an input of:-

    [; (x_1, y_1) (x_2, y_2) ... ;]

And we're trying to determine the following function:-

    [; y=f(x) ;]

* Unsupervised learning has an input of just a series of data points:-

    [; x_1, x_2, ... ;]

These data points might have many datapoints/features, and we want to learn some patterns, which we
can express as a probability distribution, e.g. the probability of a random variable being equal to
a given value:-

    [; P(X=x) ;]

* Reinforcement learning consists of a sequence of action and state transitions:-

    [; s, a, s, a ;]

Also you have a series of rewards for given states. What you're trying to do is find an optimal
policy, such that you know what the right thing to do is in a given state:-

    [; \pi ;]

Let's look at some examples and determine which form of learning is most appropriate:-

* Speech recognition - examples of voice recordings, and transcripts of intermittent text for each
  recording, trying to learn a model of language - supervised learning.

* Analysis of spectral data from stars such that we can cluster stars into particular types which
  might be of interest to astronomers - the data is a series of frequencies being transmitted for
  each star - unsupervised learning.

* Level pressing - Trying to train a rat to press a lever to get a release of food when certain
  conditions are met - reinforcement learning.

* Elevator controller - Need to decide which elevator needs to go up/down in a bank of elevators
  based on percepts, i.e. button presses in the building. Data is a series of button presses and
  wait time. We are attempting to minimise wait time - reinforcement learning.

## MDP Review ##

MDP = 'Markov Decision Processes'

We have a set of states + actions:-

    [; s \in S ;]
    [; a \in $ Actions$ (s) ;]

And a start state:-

    [; s_0 ;]

We need a transition function that says - how does the world evolve as we take actions:-

    [; P(s'|s, a ) ;]

This is a probability distribution as the world is stochastic.

This is sometimes denoted:-

    [; T(s, a, s') ;]

We then need a reward function which is sometimes applicable to the whole triple, e.g.:-

    [; R(s, a, s') ;]

And sometimes applicable only to the result state, e.g.:-

    [; R(s') ;]

Which is applicable to the 4x3 grid world + backgammon.

## Solving a MDP ##

To solve an MDP we need to find a policy:-

    [; \pi(s) ;]

We want to maximise the discounted total reward:-

    [; \sum_t \gamma^t R(s_t, \pi(s_t), s_{t+1}) ;]

We apply the discount factor, [; \gamma ;], such that future rewards are valued less than sooner
awards to prevent the process from potentially going on for too long, i.e. such that the sum total
is bounded.

If we solve the Markov process, we have a utility function as follows, which is the maximum over all
actions of the expected value:-

    [; U(s) = max_a \sum_{s'}P(s'|s, a)U(s') ;]

## Agents of Reinforcement Learning ##

What if we don't know the reward function, R, or even P, the transition model of the world? Via
reinforcement learning we can determine R and P, or use substitutes which tell you as much as you
know so you don't have to use R or P.

We have a number of choices:-

    +--------------------+-----------+-------------+-----------+
    |       Agent        |   Know    |    Learn    |    Use    |
    +--------------------+-----------+-------------+-----------+
    |Utility-Based Agent |     P     |    R->U     |     U     |
    +--------------------+-----------+-------------+-----------+
    |  Q-Learning Agent  | (nothing) |   Q(s, a)   |     Q     |
    +--------------------+-----------+-------------+-----------+
    |    Reflex Agent    | (nothing) |    pi(s)    |    pi     |
    +--------------------+-----------+-------------+-----------+

Here Q is not a utility of states, rather of state-action pairs. It tells us for any given state and
any given action, which is the utility of that result without knowing the utilities and rewards
individually.

A reflex agent is called a reflex agent because it's a pure stimulus response - I'm in a certain
state, so I take a certain action. No need to model world in terms of what are the transitions,
where am I going to go next, just go ahead and take the action.

## Passive vs. Active ##

The next choice is how adventurous the agent is going to be.

One possibility is a passive reinforcement learning agent, using any of the agent designs. It's
passive because it has a fixed policy and executes that policy, but it learns about the reward
function, R, and perhaps the transition function, P, if it doesn't know that already, by executing
the fixed policy.

As an example - imagine you're on a ship in unchartered waters, and the captain has a policy for
piloting the ship. You can't change the policy and they will execute it no matter what, and you want
to learn as much as you can about the unchartered waters, i.e. learn the reward function, using the
states and transitions that the ship is going through. You learn, and record what you learn, but
that doesn't change the policy.

The opposite is *active* reinforcement learning, e.g. let's say you've done such a great job of
learning about the unchartered water that the captain hands over control, so as you learn you can
change the policy. It's good because it means you can cash in on your reward early, and also
because it gives you a possibility to explore, so instead of asking what the best action is now, you
can determine what the best action is so that you can learn to do better in the future.

## Passive Temporal Difference Learning ##

Let's look at passive reinforcement learning. We're looking at an algorithm called 'passive temporal
difference learning', or TD, which means as we move from one state to the next we look at the
difference between the two states and learn that, then back-up the values from one state to the
next. Looking at the grid example again:-

<img src="http://codegrunt.co.uk/images/ai/10-passive-temporal-difference-learning-1.png" />

We're going to follow a fixed policy, [; \pi ;], and when we hit the +1 state it propagates to
surrounding states so we know it's good to be in nearby states.

We're trying to build up a table of utilities for each state, U(s), and keep a track of the number
of times we've reached each state in a table N(s). The table starts empty (not values set to zero,
rather actually empty) and the count starts at zero for each state.

We'll run the policy, have a trial that goes through the state, when it gets to a terminal state we
run again, and keep track of utilities and number of times we've visited each state, gradually
getting a better idea of the actual utility of each state.

The inner loop of the algorithm looks as follows:-

    if s' is new then [; U[s'] \leftarrow r' ;]
    if s is not null then
        increment [; N_s[s] ;]
        [; U[s] \leftarrow U(s) + \alpha(N_s[s])(r + \gamma U[s'] - U[s]) ;]

So, moving to the first square from the start, assuming the reward is zero, we add an entry into the
U table, and so on as we work are way through a path to the +1 terminal, assuming the policy is
good, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/10-passive-temporal-difference-learning-2.png" />

As we go we have to back-up values, so we apply the above bottom formula.

Here, [; \alpha ;] is the learning rate, i.e. a value which tells us how much we want to move the
utility towards a potentially better result. It should be such that if we're brand new, we take a
big step, however if we've seen the state several times then we'd rather take a smaller step.

If we assume:-

    [; \alpha = \frac{1}{N(s)+1} ;]

And assume that there is no discounting:-

    [; \gamma = 1 ;]

Then, considering the utility of the square immediately to the left of the +1 state we have:-

    [; U(s) \leftarrow 0 + \frac{1}{2}(0 + 1 - 0) = \frac{1}{2} ;]

On a second update we have, for the next square to the left:-

    [; U(s) \leftarrow 0 + \frac{1}{2} - 0 = \frac{1}{6} ;]

And for the square immediately to the left of the +1:-

    [; U(s) \leftarrow \frac{1}{2} + \frac{1}{3}(0 + 1 - \frac{1}{2}) = \frac{2}{3} ;]

E.g.:-

<img src="http://codegrunt.co.uk/images/ai/10-passive-temporal-difference-learning-3.png" />

## Passive Agent Results ##

Here are some results from running the passive TD algorithm on the 4x3 maze:-

<img src="http://codegrunt.co.uk/images/ai/10-passive-agent-results-1.png" />

On the right we have the average error across all states in the maze. Starts off very high, but
gradually reduces considerably, converging to around 0.05 root mean square error after 60 trials.

On the left is the utility estimates for various different states and they converge around 500
trials, however they were quite inaccurate for the first 100 or so trials, it took a while for them
to settle down.

## Weaknesses Question ##

What are the weaknesses of passive TD?

* Long convergence - long time for convergence.
* Limited by policy - restricted by the policy we don't control.
* Missing states - States we haven't visited don't get estimates.
* Poor estimate - Low-count states get a poor estimate.

## Active Reinforcement Learning ##

Let's examine a simple approach known as 'greedy' reinforcement learning.

Uses passive TD, but after each time we update the utilities, we recompute the optimal policy,
[; \pi ;].

We throw away our old policy:-

    [; \pi_1 ;]

And replace with our new policy:-

    [; \pi_2 ;]

So if a policy is flawed, the algorithm tends to move away from it towards a better policy.

## Greedy Agent Results ##

Here's the result of running the greedy agent over 500 trials:-

<img src="http://codegrunt.co.uk/images/ai/10-greedy-agent-results-1.png" />

We're graphing two things here - the error, and 'policy loss' which both jump down suddenly after a
certain number of trials (around 40-50).

Policy loss is the difference between the candidate policy and the optimal policy. The policy loss
never quite gets to zero.

You can see from the policy that it is not optimal, e.g. (2,1) is going the wrong way. Because it's
greedy it finds a good path and sticks to it regardless (e.g. the top path).

## Balancing Policy ##

How do we get the learner out of its rut? Improved for a while, but then got stuck in a sub-optimal
policy.

To find the optimal route we need to not always be taking the best route for us while we are
learning. To do this, we need to stop exploiting the best policy we know so far, and start
exploring.

Exploring could lead us astray and waste a lot of time, so we have to determine the right
trade-off - when is it worth exploring for a pay off in the long-term even if it hurts us in the
short-term?

One possibility is random exploration, i.e. following our best policy some of the time, and then
sometimes randomly take sub-optimal action. This works, but is slow to converge.

To converge, we really need to understand what's going on with exploitation vs. exploration.

## Errors in Utility Questions ##

We keep track of the optimal policy we know so far, which is continually being updated:-

    [; \not \pi $ $ \pi ;]

We keep a track of the utility of states which is also continually getting updated:-

    [; U(s) ;]

And we're keeping track of the number of times we visit each state, which gets incremented:-

    [; N_s ;]

What could go wrong? How might our utility estimates be sub-optimal? Firstly - we haven't sampled
enough, so instead get random fluctuations rather than a true estimate. Secondly, we can get a bad
utility because our policy was off - the utility isn't as high as it could be.

Looking at sampling and policy errors:-

* Both poor policy and low sampling can make U too low,
* Too low sampling can make U too high, but not bad policy, as the optimal policy has maximum
  utility, so a poor policy by definition won't make the utility too high.
* Low sampling can be improved with higher N, but poor policy isn't affected as no matter how many
  trials you use, a poor path is a poor path - you might reduce variance, but not the mean.

## Exploration Agents ##

These sources of error suggest a design for an exploration agent which will be more proactive about
exploring the world when it is uncertain, and fall back to exploiting the best policy when it's more
certain.

We take the original TD approach, modifying such that we add a large reward, possibly the maximum
reward available, for a visit count under a certain amount, e.g. our utility estimate changes to:-

    [; U(s) = +R ;]

When:-

    [; N_s < e ;]

So when the no. of visits to a state is below our exploration threshold e, we weight our utility.

## Exploration Agent Results ##

Here are the results of the exploratory agent:-

<img src="http://codegrunt.co.uk/images/ai/10-exploration-agent-results-1.png" />

It does considerably better than the passive or greedy agents. We only needed to go through 100
trials rather than 500 to get good results, so it's converging much faster and to better results. It
actually finds the exact right policy.

The error isn't quite reduced to zero, but we are still able to find the exact right policy.

## Q Learning 1 + 2 ##

Let's assume we have performed the learning and have a utility model, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/10-q-learning-1-2-1.png" />

We now have our utilities, but how do we determine a policy? We have to multiply by our transition
possibilities, e.g.:-

    [; \pi^*(s) = argmax_{a \in A(s)} \sum_{s'} P(s'|s, a)U(s') ;]

Sometimes we're given the transition model, but in other cases we don't know it, so we can't then
apply our utilities. In Q-learning, we don't know U directly, and we don't need to know the
transition model, instead we learn a direct mapping, e.g.:-

    [; \pi^*(s) = argmax_{a \in A(s)} \sum_{s'} Q(s, a) ;]

We perform Q-learning by starting off with a table of q-values, e.g. for the 4x3 grid:-

<img src="http://codegrunt.co.uk/images/ai/10-q-learning-1-2-2.png" />

There are more entries in the table than in the utility table - for each state, the elements have
been divided up by action. They all start out at q-utility 0, but as we go we update using the
following formula:-

    [; Q(s, a) \leftarrow Q(s, a) + \alpha(R(s) + \gamma Q(s', a') - Q(s, a)) ;]

Very similar to TD formula, and we back-up values in a similar fashion too. Each time we're updating
actions for each state rather than the state as a whole.

## Pacman 1 + 2 ##

In a sense we've learnt all there is to know about reinforcement learning (though of course it's a
huge field and there is a lot that hasn't been covered). What we have will work well for a small
problem like the 4x3 grid, however it won't work for a more serious problem like flying a model
helicopter or a backgammon ai as there are just too many states, and it's too hard to build up the
correct utility values or q-values.

Let's consider a simpler example - pacman:-

<img src="http://codegrunt.co.uk/images/ai/10-pacman-1-2-1.png" />

As you can see, not a great situation for pacman as he is about to get eaten by the ghosts, a
situation which could be learnt about through reinforcement learning. However it is not the case
that the following state will be considered to be the same:-

<img src="http://codegrunt.co.uk/images/ai/10-pacman-1-2-2.png" />

We need to find a generalisation which permits the learning of one set of states to be applied to
another.

We can use the same type of approach as reinforcement learning, only we look at a collection of
important features rather than considering every possible state, e.g.:-

    [; s = [ f_1, f_2, ... ] ;]

Where [; f_i ;] are individual features.

Features could be things such as the distance to the nearest ghost or the square/inverse square of
the distance, or the distance from a pill, or the number of ghosts remaining.

We could represent the q-values as follows:-

    [; Q(s, a) = \sum_i w_i \times f_i ;]

Where [; w_i ;] is a weighting given to each feature.

The task then is to find good values for the weightings.

What would make the weightings good are such that similar states have the same value, and they would
be bad should similar states have significantly different values. Bad values could be due to
ignoring an important feature, e.g. if a feature was whether we were in a tunnel, an important
feature to consider would be whether the tunnel is a dead-end or not - treating them the same
would not be a good idea.

We can make a small modification to our q-learning algorithm, e.g.:-

    [; Q(s, a) \leftarrow ... ;]
    [; w_i \leftarrow ... ;]

Here the weightings are updated in the same way the q-values would be.

It's a very similar process to supervised learning, even though the learning isn't actually
supervised here. It's as if we are bringing our own supervision to reinforcement learning.

## Conclusion ##

We've learned to do a lot with MDPs. If we don't know what the MDP is, we know how to estimate +
solve it. We can estimate the utility for a fixed policy [; \pi ;], or we can estimate the q-values
for the optimal policy while executing an exploration policy. We've seen how we can balance
exploration vs. exploitation.

Reinforcement learning is one of the most exciting areas of AI, some of the biggest surprises have
come out of it, e.g. the Backgammon player or the helicopter flier.
