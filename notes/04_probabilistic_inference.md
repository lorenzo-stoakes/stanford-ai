AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

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

The answer is going to be a complete joint probability distribution over query variables, called the
'posterior distribution' given the evidence:-

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
