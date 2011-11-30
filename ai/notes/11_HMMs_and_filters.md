AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 11 - HMMs and Filters
--------------------------

## Introduction ##

Talking about Hidden Markov Models and filter algorithms. Very important for use in robots for
example.

Very applicable in finance, medicine, etc.

## Hidden Markov Models ##

Abbreviated - HMMs

Used to:-

* Analyse time series
* Predict time series

### Applications ###

* Robotics
* Medical
* Finance
* Speech and language technologies

Many more.

At the heart of huge number of actual real-life deployed products.

Whenever there is a time series with noise or uncertainty, then using HMMs is the method of choice.

## Bayes Network of HMMs ##

The nature of HMMs are well characterised by the following Bayes network:-

<img src="http://codegrunt.co.uk/images/ai/11-bayes-network-of-hmms.png" />

These kinds of networks are the core of HMMs as well as various filters such as Kalran filters,
particle filters, etc. These sound cryptic/meaningless, however you may come across them in future.

If we examine the states, e.g.:-

    [; S_1 $ to $ S_N ;]

This is known as a Markov chain - each state only depends on its immediate predecessor.

What makes it a *hidden* Markov model/chain is that there are measurement variables. Instead of
being able to observe the state itself, you observe these (e.g. the Z's in the above diagram).

## Localisation Problem Examples ##

E.g. tour guide robot in a museum (video). We need to find out where in the world the robot is at
any given time in order to do its job. It doesn't have a sensor which actually tells it where it is,
rather it uses laser rangefinders which measure distances to surrounding objects. Also has a map of
the environment, with which it can compare rangefinder measures and thus infer where it might be.

The hidden state here is the robot's location, and the known measurement variables are the
rangefinder results. Deriving this hidden state is the problem of filtering.

This is very much applicable to the previously shown diagram, where the series of states are the
locations of the robot in the museum.

Another example is the underground mapping robot, however here the robot builds the map from
scratch. It uses a 'particle filter' to perform this mapping by considering all the possible
locations the robot might be at any given point (made uncertain by noise in the motor + sensors of
the robot), represented in the video by a number of lines mapping its progress through the mine. By
then joining the loop of where it has been, the robot is able to select some lines as more likely
than others, and by doing this over + over again it is able to build a coherent map of the mine.

One final example on speech recognition:-

<img src="http://codegrunt.co.uk/images/ai/11-localisation-problem-examples-1.png" />

If you have a microphone which records speech and have your computer recognise speech you will
likely come across Markov models.

In this example the phrase 'speech lab' is being spoken. If you 'blow up' a section of this diagram
you get an oscillation, for which the software's job is to convert back into letters. There is a lot
of variation and background noise, etc. - very challenging. There has been huge progress in this
area, and the best speech recognition software all use HMMs.

## Markov Chain Question 1 + 2 ##

Let's consider an un-hidden Markov chain.

Let's consider two types of weather - rainy (R) and sunny (S):-

<img src="http://codegrunt.co.uk/images/ai/11-markov-chain-questions-1-2-1.png" />

Where there is a 0.6 probability that it remains rainy and 0.4 chance it becomes sunny if rainy, and
a 0.8 chance of it remaining sunny and a 0.2 chance it becomes rainy if sunny.

This is clearly temporal. So let's assume at time 0:-

    [; P(R_0) = 1 ;]
    [; P(S_0) = 0 ;]

The diagram gives us:-

    [; P(R_{i+1}|R_i) = 0.6 ;]
    [; P(S_{i+1}|R_i) = 0.4 ;]
    [; P(R_{i+1}|S_i) = 0.2 ;]
    [; P(S_{i+1}|S_i) = 0.8 ;]

By the theorem of total probability we have:-

    [; P(R_{i+1}) = P(R_{i+1}|R_i)P(R_i) + P(R_{i+1}|S_i)P(S_i) ;]
    [; P(S_{i+1}) = P(S_{i+1}|R_i)P(R_i) + P(S_{i+1}|S_i)P(S_i) ;]

So:-

    [; P(R_{i+1}) = 0.6P(R_i) + 0.2P(S_i) ;]
    [; P(S_{i+1}) = 0.4P(R_i) + 0.8P(S_i) ;]


Therefore:-

    [; P(R_1) = 0.6 \times 1 + 0.2 \times 0 = 0.6 ;]
    [; P(S_1) = 0.4 \times 1 + 0.8 \times 0 = 0.4 ;]

    [; P(R_2) = 0.6 \times 0.6 + 0.2 \times 0.4 = 0.44 ;]
    [; P(S_2) = 0.4 \times 0.6 + 0.8 \times 0.4 = 0.56 ;]

    [; P(R_3) = 0.6 \times 0.44 + 0.2 \times 0.56 = 0.376 ;]
    [; P(S_3) = 0.4 \times 0.44 + 0.8 \times 0.56 = 0.624 ;]

Consider another situation:-

<img src="http://codegrunt.co.uk/images/ai/11-markov-chain-questions-1-2-2.png" />

Hence:-

    [; P(A_{i+1}|A_i) = 0.5 ;]
    [; P(B_{i+1}|A_i) = 0.5 ;]
    [; P(A_{i+1}|B_i) = 1 ;]
    [; P(B_{i+1}|B_i) = 0 ;]

We assume:-

    [; P(A_0) = 1 ;]

Hence:-

    [; P(B_0) = 0 ;]

Again, from total probability we have:-

    [; P(A_{i+1}) = P(A_{i+1}|A_i)P(A_i) + P(A_{i+1}|B_i)P(B_i) ;]
    [; P(B_{i+1}) = P(B_{i+1}|A_i)P(A_i) + P(B_{i+1}|B_i)P(B_i) ;]

Thus:-

    [; P(A_{i+1}) = 0.5P(A_i) + P(B_i) ;]
    [; P(B_{i+1}) = 0.5P(A_i) ;]

So:-

    [; P(A_1) = 0.5 \times 1 + 0 = 0.5 ;]
    [; P(B_1) = 0.5 \times 1 = 0.5 ;]

    [; P(A_2) = 0.5 \times 0.5 + 0.5 = 0.75 ;]
    [; P(B_2) = 0.5 \times 0.5 = 0.25 ;]

    [; P(A_3) = 0.5 \times 0.75 + 0.25 = 0.625 ;]
    [; P(B_3) = 0.5 \times 0.75 = 0.375 ;]

## Stationary Distribution + Question ##

Looking at the situation given in the last problem again:-

<img src="http://codegrunt.co.uk/images/ai/11-stationary-distribution-question-1.png" />

What if we go to very long time, e.g.:-

    [; P(A_{1000}) ;]

Or in the limit, e.g.:-

    [; P(A_\infty) = \lim_{t \uparrow \infty} P(A_t) ;]

Essentially going to the limit means we wait for a long time.

This latter probability is known as the *stationary distribution*. And every Markov chain settles to
a stationary distribution (or sometimes a limit cycle if the transitions are deterministic - which
we don't care about).

The secret to determining this distribution is to realise that, in the limit:-

    [; P(A_t) = P(A_{t-1}) = P(A_\infty) ;]

We know that:-

    [; P(A_t) = P(A_t|A_{t-1})P(A_{t-1}) + P(A_t|B_{t-1})P(B_{t-1}) ;]

From total probability.

If we call the stationary distribution value (which we want) 'X', e.g.:-

    [; X = P(A_\infty) ;]

Then we can express the formula we derived from total probability as:-

    [; X = P(A_t|A_{t-1})X + P(A_t|B_{t-1})(1 - X) ;]

Plugging in values:-

    [; X = 0.5X + 1 - X ;]

Thus:-

    [; 1.5X = 1 ;]
    [; X = \frac{2}{3} ;]

And thus:-

    [; P(B_\infty) = \frac{1}{3} ;]

It's still a Markov chain, so we're still flipping between A and B, however now have an idea of
frequencies of each - i.e. two thirds of the time we're in A and the other third of the time we're
in B.

Let's examine the rain problem again:-

<img src="http://codegrunt.co.uk/images/ai/11-stationary-distribution-question-2.png" />

We have:-

    [; X = P(R_t|R_{t-1})X + P(R_t|S_{t-1})(1 - X) ;]

Thus:-

    [; X = 0.6X + 0.2(1 - X) ;]
    [; 0.6X = 0.2 ;]

Hence

    [; X = P(R_\infty) = \frac{1}{3} ;]

## Finding Transition Probabilities + Question ##

You can determine the transition probabilities of a Markov chain like the following:-

<img src="http://codegrunt.co.uk/images/ai/11-finding-transition-probabilities-question-1.png" />

By observation. You want to determine the missing probabilities here, based on observation of the
following values:-

    RSSSRSR

Using maximum likelihood, we have:-

    [; P(R_0) = 1 ;]

Since we started on a rain day.

If we consider transitions, then we can simply count instances of transitioning from sunny to sunny, sunny to rainy, rainy to sunny, rainy to rainy e.g.:-

    [; P(S|S) = \frac{2}{4} = 0.5 ;]
    [; P(R|S) = \frac{2}{4} = 0.5 ;]
    [; P(S|R) = \frac{2}{2} = 1 ;]
    [; P(R|R) = \frac{0}{2} = 0 ;]

E.g.:-

    SSSSSRSSSRR

    [; P(R_0) = 0 ;]
    [; P(S|S) = \frac{6}{8} = \frac{3}{4} ;]
    [; P(R|S) = \frac{2}{8} = \frac{1}{4} ;]
    [; P(S|R) = \frac{1}{2} ;]
    [; P(R|R) = \frac{1}{2} ;]

## Laplacian Smoothing Question ##

One of the problems with HMMs is overfitting - e.g. our prior probability is always 1 for the state
of the first day, e.g. [; P(R_0) = 1 ;].

Consider:-

    RSSSS

There are 2 possible states in each case (R and S), and k = 1, so given our LaPlacian smoothing
algorithm, we have:-

    [; P(x) = \frac{count(x) + k}{N + k|x|} ;]
    [; P(x) = \frac{count(x) + 1}{N + 2} ;]

In terms of ...

    [; P(R_0) = \frac{2}{3} ;]

There are 3 transitions from S, and 1 from R, so the denominator is going to be 5 and 3
respectively.

There are 3 transitions from S to S so:-

    [; P(S|S) = \frac{4}{5} ;]

There are 0 transitions from S to R so:-

    [; P(R|S) = \frac{1}{5} ;]

There is 1 transition from R to S so:-

    [; P(S|R) = \frac{2}{3} ;]

There are 0 transitions from R to R so:-

    [; P(R|R) = \frac{1}{3} ;]

## HMM Happy Grumpy Problem + Question ##

Looking at rainy/sunny problem again:-

<img src="http://codegrunt.co.uk/images/ai/11-hmm-happy-grumpy-problem-question-1.png" />

We assume the following:-

    [; P(R_0) = \frac{1}{2} ;]
    [; P(S_0) = \frac{1}{2} ;]

What makes this a *hidden* Markov model is that we don't get to observe the rainy/sunny variable,
rather we get to observe something else - in this case whether the subject is happy/grumpy based on
the weather, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/11-hmm-happy-grumpy-problem-question-2.png" />

Let's assume happy on day 1, e.g.:-

    [; H_1 ;]

What's the posterior probabilty of rain on day 1? E.g.:-

    [; P(R_1|H_1) = \frac{P(H_1|R_1)P(R_1)}{P(H_1)} ;]
    [; P(R_1) = P(R_1|R_0)P(R_0) + P(R_1|S_0)P(S_0) ;]

    [; = 0.6 \times 0.5 + 0.2 \times 0.5 = 0.4 ;]

Plugging in the values we get:-

    [; P(R_1|H_1) = 0.229 ;]

We've determined that the marginal probability of it being rainy on day 1 is 0.4, so by determining
that the subject is happy, we significantly reduce the probability - this is because when the
subject is happy it is far more likely to be sunny.

Great example of the power of Baye's rule in a relatively complicated HMMs.

## Wow You Understand ##

Professor Thrun goes mental at what we understand. We are now good at:-

* Prediction - Predict the next state/measurement.
* State Estimation - fancy way of saying able to compute a probability for the internal or hidden
  state given measurements.

## HMMs and Robot Localisation ##

<img src="http://codegrunt.co.uk/images/ai/11-hmms-and-robot-localisation-1.png" />

Video of HMMs used for robot navigation. This is a toy robot in a grid world (consists of discrete
cells). Knows where north is at all times, can sense left, right, up and down. Can determine whether
a wall is in a particular direction in the adjacent cell.

Initially has no clue where it is - known as a 'global localisation problem'. Uses its sensors and
actuators to localise itself.

In the first instance, it senses walls north and south but not west and east, which changes the
probabilities:-

<img src="http://codegrunt.co.uk/images/ai/11-hmms-and-robot-localisation-2.png" />

The posterior probability have now increased for places where there is a wall north and south, and
decreased where this is not the case.

The lighter grey shaded areas have a lower probability because this takes into account there being
exactly one sensing error in the south sensor. Errors are less likely than not having an error thus
this is a lighter shade than other north/south walled cells.

If we move right, the probabilities change again:-

<img src="http://codegrunt.co.uk/images/ai/11-hmms-and-robot-localisation-3.png" />

Probabilities have decayed, the leftmost cell has reduced probability because though it was
consistent before, it is less consistent with moving right and measuring a wall to the
north/south. The only ones which are consistent are the three immediately next to the robot.

As we keep on moving the probabilities change again:-

<img src="http://codegrunt.co.uk/images/ai/11-hmms-and-robot-localisation-4.png" />

<img src="http://codegrunt.co.uk/images/ai/11-hmms-and-robot-localisation-5.png" />

In this final state it has localised itself.

## HMM Equations ##

Looking at the maths of HMMs.

## HMM Localisation Example ##

## Particle Filters ##

## Localisation and Particle Filters ##

## Particle Filter Algorithm ##

## Particle Filter Pros and Cons ##

## Conclusion ##
