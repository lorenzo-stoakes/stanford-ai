AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 20 - Robotics II
---------------------

## Prediction ##

Coming back to Monte Carlo localisation - consider moving from position (x, y) with heading
[; \theta ;], moving with velocity v and angular velocity [; \omega ;], we end up in a definite
location as determined by these values. However, if we introduce noise, then we end up with several
different potential positions after a prediction step:-

<img src="https://ljs.io/img/ai/20-prediction-1.png" />

## Measurement Question ##

Let's consider particles which are either on a lane marker or not, and the robot has a sensor which
determines whether the road surface is bright or not.

We also have:-

    P(bright | on lane marker ) = 0.8
    P(dark   | off lane marker) = 0.9

This is non-trivial.

In hand-wavey terms - The weighting for each on-lane marking particle is 0.8, and the weighting for
each off-line marking particle is 0.1 (1-0.9 since we know the sensor is reading bright). If we now
multiply by the number of particles on each surface, we end up with:-

    [; 3 \times 0.8 = 2.4 ;]
    [; 3 \times 0.1 = 0.3 ;]

Which gives us a total of 2.7.

If we then normalise, we end up with:-

    [; w(x^{[1]}) = \frac{8}{27} ;]
    [; w(x^{[2]}) = \frac{1}{27} ;]

We need the probabilities of each item to add up to 1.

## Resampling Question ##

The next step in the particle filter algorithm is to resample, with on-lane marking particles at
probability 0.2963, and off-line marking particles at 0.037 as determined in the previous question:-

<img src="https://ljs.io/img/ai/20-resampling-question-1.png" />

We then pick new particles, weighted by these probabilities:-

<img src="https://ljs.io/img/ai/20-resampling-question-2.png" />

Which are then skewed by movement (with noise):-

<img src="https://ljs.io/img/ai/20-resampling-question-3.png" />

The process is:-

* Look at measurement
* Compute weights
* Sample
* Predict

## Planning Question ##

A key problem is that the robots have to decide what to do next. Will address this at several levels
of abstraction, starting with a silly one.

Consider the following world:-

<img src="https://ljs.io/img/ai/20-planning-question-1.png" />

We can determine the value at source using value iteration.

## Road Graph ##

We apply the exact same thing to a real-world problem:-

<img src="https://ljs.io/img/ai/20-road-graph-1.png" />

Here we are taking heading into account as well as distance from the goal.

The reason the section of road beneath the mission goal is coloured green is that we expect the car
to be pointing north when it reaches the goal. The map is taking into account one-way systems, etc.

## Cost Question ##

Consider the following problem:-

<img src="https://ljs.io/img/ai/20-cost-question-1.png" />

Need to determine the maximum cost of left turn such that we never turn left - got to make it not
worth it.

## Dynamic Programming 1 + 2 ##

By assigning weights according to what you do/do not want the robot to do you can have a major
impact on the route an automated car takes. Essentially dynamic programming.

## Robotic Path Planning ##

This is a rich field, can't give a complete survey, too much to cover.

Consider how you'd use A*:-

<img src="https://ljs.io/img/ai/20-robotic-path-planning-1.png" />

Here we've discrete-ised the state space and determined a path. This is not really appropriate for a
car, however, as a car cannot make such sharp turns.

The fundamental issue is that A* is discrete, but the world continuous. The question is - is there a
version of A* which can deal with a continuous environment and provide provably executable paths?
This is a big question in robot motion planning.

The key to solving this relates to the state transition function.

Consider attempting to solve this using a series of very small step simulations using the equations
we obtained before:-

    [; x' = x + v \Delta t \cos \theta ;]
    [; y' = y + v \Delta t \sin \theta ;]
    [; \theta' = \theta + \Delta t \omega ;]

Consider the following:-

<img src="https://ljs.io/img/ai/20-robotic-path-planning-2.png" />

Here the 'hybrid A*' algorithm stores x', y' and [; \theta' ;] for the cell. Note that once the cell
has been expanded these values do not change and thus if you come back into the cell from another
source, with different x', y', and [; \theta' ;] values as a result, these will not actually get
picked up correctly as it doesn't bother recalculating these.

This leads to a lack of completeness, but does provide for correctness insofar as the equations give
valid results (to the degree of accuracy that they provide).

## Path Planning Examples ##

Here's an example of applying the hybrid A* algorithm to a real situation:-

<img src="https://ljs.io/img/ai/20-path-planning-examples-1.png" />

There are obviously different variations of A*, but can't go into vast detail here again due to
limited scope. In general though, despite there being some sophistications going on which haven't
been covered due to scope, A* is the core of how real robot navigation works.

## Conclusion ##

Finished the short overview, but we have still covered many examples, e.g.:-

* Perception via particle filters
* Planning via MDP, A*
