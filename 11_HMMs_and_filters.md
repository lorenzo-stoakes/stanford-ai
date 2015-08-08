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

<img src="https://ljs.io/img/ai/11-bayes-network-of-hmms.png" />

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

<img src="https://ljs.io/img/ai/11-localisation-problem-examples-1.png" />

If you have a microphone which records speech and have your computer recognise speech you will
likely come across Markov models.

In this example the phrase 'speech lab' is being spoken. If you 'blow up' a section of this diagram
you get an oscillation, for which the software's job is to convert back into letters. There is a lot
of variation and background noise, etc. - very challenging. There has been huge progress in this
area, and the best speech recognition software all use HMMs.

## Markov Chain Question 1 + 2 ##

Let's consider an un-hidden Markov chain.

Let's consider two types of weather - rainy (R) and sunny (S):-

<img src="https://ljs.io/img/ai/11-markov-chain-questions-1-2-1.png" />

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

<img src="https://ljs.io/img/ai/11-markov-chain-questions-1-2-2.png" />

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

<img src="https://ljs.io/img/ai/11-stationary-distribution-question-1.png" />

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

<img src="https://ljs.io/img/ai/11-stationary-distribution-question-2.png" />

We have:-

    [; X = P(R_t|R_{t-1})X + P(R_t|S_{t-1})(1 - X) ;]

Thus:-

    [; X = 0.6X + 0.2(1 - X) ;]
    [; 0.6X = 0.2 ;]

Hence

    [; X = P(R_\infty) = \frac{1}{3} ;]

## Finding Transition Probabilities + Question ##

You can determine the transition probabilities of a Markov chain like the following:-

<img src="https://ljs.io/img/ai/11-finding-transition-probabilities-question-1.png" />

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

<img src="https://ljs.io/img/ai/11-hmm-happy-grumpy-problem-question-1.png" />

We assume the following:-

    [; P(R_0) = \frac{1}{2} ;]
    [; P(S_0) = \frac{1}{2} ;]

What makes this a *hidden* Markov model is that we don't get to observe the rainy/sunny variable,
rather we get to observe something else - in this case whether the subject is happy/grumpy based on
the weather, e.g.:-

<img src="https://ljs.io/img/ai/11-hmm-happy-grumpy-problem-question-2.png" />

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

<img src="https://ljs.io/img/ai/11-hmms-and-robot-localisation-1.png" />

Video of HMMs used for robot navigation. This is a toy robot in a grid world (consists of discrete
cells). Knows where north is at all times, can sense left, right, up and down. Can determine whether
a wall is in a particular direction in the adjacent cell.

Initially has no clue where it is - known as a 'global localisation problem'. Uses its sensors and
actuators to localise itself.

In the first instance, it senses walls north and south but not west and east, which changes the
probabilities:-

<img src="https://ljs.io/img/ai/11-hmms-and-robot-localisation-2.png" />

The posterior probability have now increased for places where there is a wall north and south, and
decreased where this is not the case.

The lighter grey shaded areas have a lower probability because this takes into account there being
exactly one sensing error in the south sensor. Errors are less likely than not having an error thus
this is a lighter shade than other north/south walled cells.

If we move right, the probabilities change again:-

<img src="https://ljs.io/img/ai/11-hmms-and-robot-localisation-3.png" />

Probabilities have decayed, the leftmost cell has reduced probability because though it was
consistent before, it is less consistent with moving right and measuring a wall to the
north/south. The only ones which are consistent are the three immediately next to the robot.

As we keep on moving the probabilities change again:-

<img src="https://ljs.io/img/ai/11-hmms-and-robot-localisation-4.png" />

<img src="https://ljs.io/img/ai/11-hmms-and-robot-localisation-5.png" />

In this final state it has localised itself.

## HMM Equations ##

So far we've been a bit hand wavy about things so let's get a bit more formal look at the
mathematics of HMMs.

We know that we are dealing with a Markov chain like this:-

<img src="https://ljs.io/img/ai/11-hmm-equations-1.png" />

Here the past, present, and future are all conditionally independent given [; X_2 ;].

Let's you efficiently perform inference.

If we consider a single measurement from a single hidden variable:-

<img src="https://ljs.io/img/ai/11-hmm-equations-2.png" />

Then we have from Baye's rule:-

    [; P(X_1|Z_1) = \frac{P(Z_1|X_1)P(X_1)}{P(Z_1)} ;]

We can look at the proportionality of this:-

    [; P(X_1|Z_1) \alpha P(Z_1|X_1)P(X_1) ;]

This is the measurement update of a HMM. Keep in mind that it needs to be normalised as previously
discussed.

We also have 'prediction' (not necessarily prediction per se, a historical term related to the fact that we might want to determine the probability for X2 given the probability for X1):-

<img src="https://ljs.io/img/ai/11-hmm-equations-3.png" />

Which is represented by the following formula:-

    [; P(X_2) = \sum_{X_1} P(X_1)P(X_2|X_1) ;]

These two equations form the maths of a HMM, and the parameters of a HMM are defined as:-

    [; P(X_2|X_1) ;]
    [; P(Z_1|X_1) ;]
    [; P(X_0) ;]

## HMM Localisation Example ##

Let's consider the application of an HMM to a real robot localisation example.

Consider 1d world where robot is lost:-

<img src="https://ljs.io/img/ai/11-hmm-localisation-example-1.png" />

The graph underneath is a histogram which shows all possible states. We bin the world into small
bins + for each bin we assign a small numerical probability for the robot being in that position.

When all the bins are of the same height, there is maximum uncertainty as to where the robot is.

Now the robot senses that it is next to a door:-

<img src="https://ljs.io/img/ai/11-hmm-localisation-example-2.png" />

The red graph is the probability of seeing a door for different locations in the environment.

We then apply Baye's rule and multiply the prior with the measurement probability to obtain the
posterior - this gives us our measurement update.

Things progress as the robot moves right:-

<img src="https://ljs.io/img/ai/11-hmm-localisation-example-3.png" />

This is the next state prediction part of the process, also known as the 'convolution' or 'state
transition' part of the process where the little bumps get shifted along with the robot, and they
are flattened out a little due to the fact that robot motion has used uncertainty.

Really a simple operation - you shift the values right, and smooth them out a little to take into
account the control noise in the robot's actuators.

Now the robot senses again:-

<img src="https://ljs.io/img/ai/11-hmm-localisation-example-4.png" />

Here we see that there is a multiplication effect going on - there is a non-uniform prior,
multiplied by a measurement - if you look at the graph, the only place where you have a high prior
probability and a high measurement probability is where the peak occurs at the second door.

Really a simple algorithm, essentially - measurements are multiplications, motion is convolution
(shifts with added noise).

## Particle Filters ##

Previous section a great segue into particle filters - one of the most successful algorithms in
AI/robots.

We're still dealing with robot localisation. The below image is data from a real robot with sensor
data. The robot is lost in the building and has range sensors (measuring distances to nearby
obstacles), and its task is to determine where it is:-

<img src="https://ljs.io/img/ai/11-particle-filters-1.png" />

The robot needs to move along the black line, but it needs to determine where it is.

The key thing with particle filters is the representation of belief. Previously we had discrete
worlds (like the sun/rain example or the small bins example). Particle filters have a very different
representation - they represent the space with a series of points/particles.

Each of the dots is a hypothesis of where the world might be, it's a concrete (x, y) value and its
heading direction - it's a 3 value vector.

The sum of all the vectors forms the belief space. The particle filter approximate a posterior via
many guesses, and the density of the guesses represent the posterior probability of being in a given
location.

Video shown, demonstrating that the particles soon cluster, with symmetry causing a duplicate in a
symmetrical location (e.g. a corridor) before the robot enters a far less symmetric room where the
symmetrical false clustering dies out.

Intuitively, each particle is a representation of a possible state, and the more consistent the
particle with a measurement, the more the sonar measurement fits into the place where the particle
says the robot is, the more likely it is to survive. This is the essence of particle filters - uses
many particles to represent a belief, and lets the particles survive in proportion to the
measurement probability, which here is the consistency of sonar range measurements with the map of
the environment given the location of the particle.

Can be implemented in less than 10 lines of (C) code.

## Localisation and Particle Filters ##

Here's our 1d localisation optimisation problem once again, this time with particle filters:-

<img src="https://ljs.io/img/ai/11-localisation-and-particle-filters-1.png" />

The particles start out spread out relatively uniformly. Going to use this example to explain every
step.

<img src="https://ljs.io/img/ai/11-localisation-and-particle-filters-2.png" />

The robot senses the door, then copies the particles over verbatim, then gives them a 'weight'. This
is called the 'importance weight'. This is equal to the measurement probability. The height of the
particle indicates the weights of the particles.

The robot now moves:-

<img src="https://ljs.io/img/ai/11-localisation-and-particle-filters-3.png" />

Here we've performed 'resampling', an algorithm which works by picking a particle from the previous
set, picking more frequently in proportion to the importance weight. The particles are then skewed
by the movement. This is the 'forward prediction' step.

We then perform a measurement step:-

<img src="https://ljs.io/img/ai/11-localisation-and-particle-filters-4.png" />

As before we've multiplied the measurement probability by the particles. Particles are starting to
clump towards the correct location.

The nice aspect of this is that particle filters work in continuous spaces, and (often
underappreciated) they use your computational resources in proportion to how likely something is, so
they use your resources in a very intelligent way.

We resample again:-

<img src="https://ljs.io/img/ai/11-localisation-and-particle-filters-5.png" />

## Particle Filter Algorithm ##

The particle filter algorithm is defined as:-

    Algorithm Particle Filter(S, U, Z)

Where:-

* S - set of particles with associated importance weights
* U - control vector
* Z - measurement vector

This constructs a new particle set S':-

    [; S' = \theta ;]
    [; \eta = 0 ;]

Where [; \eta ;] is our normalisation factor as 

The steps are:-

    For i=1...n
        Sample j ~ {w} with replacement

Here we sample an index j according to the index weights associated with the input particle set. The
probability of picking the particle is exactly the importance weight, w.

        [; x' ~ p(x'|u, s_j) ;]
        [; w' = p(z|x') ;]

Here we determine the new particle, x, and the specific particle in question [; s_j ;], and its
associated importance weight.

        [; s' = s' \cup \{ <x', w'> \} ;]

Here we add the new particle into our new particle set.

        [; \eta = \eta + w' ;]

Here we increment our normalisation factor by our new importance weight.
        
    end
    For i = 1 ... n
        [; w_i = \frac{1}{\eta}w_i ;]
    end

Here we normalise.

Really a very simple algorithm + easy to implement, certainly when compared to implementing a Bayes
network for example.

## Particle Filter Pros and Cons ##

Particle filters are:-

Pros

* Easy to implement.
* Work well in many applications - the self-driving cars use particle filters for localisation,
  mapping and a number of other functions. They work well due to being easy to implement,
  computationally efficient and can deal with highly non-monotonic + v. complex posterior
  distribution with many peaks - this is important, many other filters cannot. Often the method of
  choice for determining an estimation method quickly for problems where the posterior is complex.

Cons

* Don't work in high-dimensional spaces - number of particles required to fill a high-dimensional
  space tends to grow exponentially with the dimensions of the space. However, there are extensions
  which can help with this with fancy names like 'Rao-Blackwellised particle filters'
* Problems with degenerate conditions, e.g. if you only have 1 or 2 particles, or no noise in your
  measurement model or control (relied upon to mix things up a little). If you have no noise you
  have to deviate from the basic paradigm.

## Conclusion ##

Learnt a lot about HMMs and particle filters. Particle filters are the most used algorithm for
interpreting sensor data. Applicable in a wide array of applications including finance, medicine,
behavioural studies, times series analysis, language technologies - basically anything involving
time and sensors/uncertainty.
