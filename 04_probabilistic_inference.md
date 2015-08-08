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

<img src="https://ljs.io/img/ai/4-overview-and-example-1.png" />

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

<img src="https://ljs.io/img/ai/4-speeding-up-enumeration-1.png" />

If all n variables are boolean variables.

In the alarm network shown earlier, we were careful to ensure that we put all the independence
relations in the structure of the network, but if we put the nodes in a different order we'd have a
different structure, e.g.:-

<img src="https://ljs.io/img/ai/4-speeding-up-enumeration-2.png" />

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

<img src="https://ljs.io/img/ai/4-variable-elimination-1.png" />

Where

* R - is it raining?
* T - is there traffic?
* L - late for next appointment?

    [; P(+l) = \sum_r \sum_t P(r)R(t|r)P(+l|t) ;]

The elimination:-

<img src="https://ljs.io/img/ai/4-variable-elimination-2.png" />

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

<img src="https://ljs.io/img/ai/4-approximate-inference-1.png" />

Here we sample by simply running the experiment over and over and looking at the counts. If the
counts are low, then random variation can cause the results to be unreliable. However, as we add
more samples the counts we get come closer to the true distribution.

Sampling has an advantage over inference in that we have a procedure for coming up with at least an
approximate value for the joint probability distribution, rather than exact inference where the
computation might be v. complex.

Another advantage is that, if we don't know what the conditional probability tables are, we can
still proceed with sampling, unlike in exact inference where we simply couldn't.

## Sampling Example ##

[YouTube Link 1](http://www.youtube.com/watch?feature=player_embedded&v=mXgfRvRmDFI)
[YouTube Link 2](Explanation)](http://www.youtube.com/watch?feature=player_embedded&v=K1ZyqpTJPK0)
[AI Class Link](https://www.ai-class.com/course/video/quizquestion/80)

Here's a network we can use to investigate how sampling can be used to do inference:-

<img src="https://ljs.io/img/ai/4-sampling-example-1.png" />

We have 4 booleans variables - each indicating whether the condition in question is the case or not.

We start the analysis with the variable which has all of its parents defined. Initially this is the
cloudy variable.

We use a random variable to determine P(C), before using the outcome of this to decide P(S|C) and
P(R|C) before, in turn, determining the value of P(W|S,R).

## Approximate Inference 2 ##

[YouTube Link](http://www.youtube.com/watch?feature=player_embedded&v=fChe7bVEdHQ)
[AI Class Link](https://www.ai-class.com/course/video/videolecture/41)

The probability of sampling a particular variable depends on the values of the parents, but they are
chosen according to the conditional probability tables, so in the limit the count of each sampled
probability will approach the true probabilities, e.g. with an infinite number of samples, this
model will provide the true joint probabilities.

We say that the sampling method is *consistent*.

We can use this kind of sampling to determine the complete joint distribution, or we can use it to
compute an individual variable.

What if we wanted to compute a conditional variable, e.g. [; P(W|\lnot C) ;]?

If you consider the sample (resulting from random variables) of +c, -s, +r, +w, then it's obvious
this is not consistent with the required conditional probability. So we *reject* this sample.

The technique of ignoring any samples which do not match the conditional probabilities which we are
interested in (but keeping the ones we are interesting in) is known as 'rejection sampling'. This procedure is also consistent.

## Rejection Sampling ##

[YouTube Link](http://www.youtube.com/watch?feature=player_embedded&v=9IdjpH4xkGM)
[AI Class Link](https://www.ai-class.com/course/video/videolecture/42)

There's a problem with rejection sampling - you end up rejecting *a lot* of the samples.

If we consider the burglary/alarm network again:-

<img src="https://ljs.io/img/ai/4-rejection-sampling-1.png" />

Say we're interested in [; P(B|+A) ;]. The problem is that burglaries are very infrequent, so you
have to put up with a large number of -b, -a cases before you get to any +b cases.

We introduce a technique to deal with this - *likelihood weighting*. This generates samples such
that we can keep everything. We fix the evidence variables (here A), so A is always positive, and
then sample the rest of the variables.

The problem with this is that the resultant set of samples is *inconsistent*. We can fix this however, by assigning a probability to each sample and weighting them appropriately.

## Likelihood Weighting + 1-2 ##

In likelihood weighting we collect samples as we did before, and add a probabilistic weight to each
sample.

Consider our cloudy/sprinkler/rain/wet grass network again:-

[Youtube Link 1](http://www.youtube.com/watch?feature=player_embedded&v=GYcIruSqT_k)
[YouTube Link 2 (explanation)](http://www.youtube.com/watch?feature=player_embedded&v=hvIL_fFvUGM)
[Youtube Link 3](http://www.youtube.com/watch?feature=player_embedded&v=jKcp0uQ_rUo)
[Youtube Link 4](http://www.youtube.com/watch?feature=player_embedded&v=ngGCGaIEvBU)

[AI Class Link 1](https://www.ai-class.com/course/video/quizquestion/81)
[AI Class Link 2](https://www.ai-class.com/course/video/videolecture/43)
[AI Class Link 3](https://www.ai-class.com/course/video/videolecture/44)

<img src="https://ljs.io/img/ai/4-likelihood-weighting-1-2-1.png" />

Assuming we are after [; P(R|+S, +W) ;], and our random variables grant cloudy positive and rain
positive, then we have a weight of 0.1 from the assumed positive sprinkler, and 0.99 weight from the
wet grass, so our overall weight is [; 0.1 \times 0.99 = 0.099 ;].

Now we are applying this weighting, the sampling is now consistent.

A problem here is that unconstrained variables might end up such that weightings end up being low,
e.g. consider cloudiness and sprinkler/rain being positive.

## Gibbs Sampling ##

[YouTube Link](http://www.youtube.com/watch?feature=player_embedded&v=QaojSzk7Hpw)
[AI Class Link](https://www.ai-class.com/course/video/videolecture/45)

Gibbs sampling takes all the evidence into account, not just the upstream evidence. It uses a method
called Markov Chain Monte Carlo (MCMC), e.g.:-

<img src="https://ljs.io/img/ai/4-gibbs-sampling-1.png" />

Using this method, we resample one non-evidence variable at a time using the values of all the other
variables.

We end up walking the space of assignment of variables, randomly.

In rejection and likelihood sampling, each sample was independent of one another. Here there is not
true. Adjacent samples vary only in one place. Regardless, the technique is still consistent.

## Monty Hall Problem ##

[YouTube Link 1](http://www.youtube.com/watch?feature=player_embedded&v=6uF6Fh0qpV0)
[YouTube Link 2 (explanation)](http://www.youtube.com/watch?feature=player_embedded&v=x7x6nHvQEQ4)
[AI Class Link](https://www.ai-class.com/course/video/quizquestion/70)

This problem concerns a game show where you have 3 doors, behind 1 of which resides an expensive
sports car, and behind the other 2 reside goats.

Once you make a choice, e.g. door 1, the host will open one of the doors, knowing the door he opens
contains a goat. He then gives you the choice between switching.

The probability of winning when sticking with door 1 is 1/3, whereas when you switch to door 2, the
probability is 2/3.

This is counter-intuitive.

There are two possibilities of course, however we have learnt from probability that just because
there are two possibilities it does not mean they are equally likely.

It's easier to say why the first door possesses 1/3 probability - when the game started each door
has 1/3 probability of hiding the car. One possible explanation is that the probabilities have to
sum to 1, so you have no choice but to assign 2/3 probability to the remaining other door. Why does
this apply to door 2 and not 1? That's because the host revealing the goat has *updated* the
probability of the door as he chose not to open it. We haven't learnt anything about door no. 1 as
opening that door was never an option for the host.

## Monty Hall Letter ##

[YouTube Link](http://www.youtube.com/watch?feature=player_embedded&v=CIrfGiP65UI)
[AI Class Link](https://www.ai-class.com/course/video/videolecture/46)

Letter from Monty Hall himself about use of his name to discuss the Monty Hall problem in which he
demonstrates that even Monty Hall does not understand the Monty Hall problem :-)
