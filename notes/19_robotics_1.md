AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 19 - Robotics I
--------------------

## Autonomous Vehicle Intro 1 + 2 ##

Thrun has been working on self-driving cars most of his professional life. Can make driving
significantly safer, lose over 1M+ people per year to traffic accidents. Can be avoided by making
cars safer. Can drive blind people/disabled people/young children/etc./all of us so we don't have to
focus on driving.

After the first DARPA grand challenge 2004 - task was to drive through 141 miles of very challenging
terrain from Nevada to California - no team made it past 5% of the way that year.

Entered (+ ultimately won) the following year with 20 students (CS294: Experiments in AI) - modified
VW Touareg using sensors on roof and actuators to activate pedals and steering pedal.

<img src="http://codegrunt.co.uk/images/ai/19-autonomous-vehicle-intro-1-2-1.png" />

Shortly before challenge was able to handle desert terrain well including steep inclines/declines,
avoiding obstacles, etc.

Using lasers to map terrain:-

<img src="http://codegrunt.co.uk/images/ai/19-autonomous-vehicle-intro-1-2-2.png" />

Went on to win DARPA grand challenge. Then created 'junior' the follow up which competed in the
DARPA urban challenge.

Used range vision laser to map terrain and particle filters + histogram filters relative to a given
map of the environment for localisation. The map was essential for navigating safely in traffic. Was
able to detect other cars using particle filters and estimate where they were, how fast they were
moving and their size.

Google car - driven over 100k miles.

In this class we'll talk about how to develop a self-driving car :-)

## Robotics Introduction + Question ##

Essentially - applying AI techniques to the problem of robotics, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/19-robotics-introduction-question-1.png" />

Robotics is:-

* Partially observable
* Continuous
* Stochastic
* Adversarial - can be argued both ways. Shouldn't think that way for driving :)

## Kinematic Question 1 + 2 ##

Discussing perception - we get sensor data and trying to estimate internal state such that it is
sufficient to determine what to do next, usually a recursive method called a filter:-

<img src="http://codegrunt.co.uk/images/ai/19-kinematic-question-1-2-1.png" />

Consider a mobile robot which is round and lives on a plane.

Kinematic state - care about where it is but not how fast it's going. 3 dimensions - (x, y, theta) -
where theta is heading.

Considering a car - same answer.

## Dynamic Question ##

To encode dynamic state need 5 dimensions - the 3 kinematic variables, as well as forward velocity
and yaw rate - the rate at which the car is turning.

## Helicopter Question 1 + 2 ##

Consider a helicopter's dimensionality - 6 kinematic variables - coordinates + roll, pitch, yaw.

Dynamic dimensionality - 12 - 6 kinematic variables + velocity in each = 12 - dynamic state of free
flying object.

## Localisation ##

Discuss localisation of a car like Junior. Uses probabilistic mapping to determine where it is -
could use GPS, however that wouldn't be effective as it often has enormous errors of 5-10m which is
clearly unsafe for driving. Localising using particle filters/histogram filters gives us ~10cm
error.

## Monte Carlo Localisation ##

Particle filters for localisation = Monte Carlo localisation. Consider a series of points (x, y,
theta), e.g:-

<img src="http://codegrunt.co.uk/images/ai/19-monte-carlo-localisation-1.png" />

Consider a 'differential wheel robot' which has independent control of two wheels - can drive both
forward, however if one moves faster than another it turns. How do we apply a particle filter to a
robot of this simplicity? Simpler than a car but not *that* much simply - comparable complexity.

We need to determine:-

    [; x' ;]
    [; y' ;]
    [; \theta ;]

Given:-

    [; \Delta t ;]
    [; v ;]
    [; \omega ;]

Where v is velocity and [; \omega ;] is turning velocity.

I.e. time step. This gives us:-

    [; x' = x + v \Delta t \cos \theta ;]
    [; y' = y + v \Delta t \sing \theta ;]
    [; \theta' = \theta + \omega \Delta t ;]

Nice equations to model relatively complex mobile robots. Simple geometry.

Robot moves on a fixed straight trajectory for [; \Delta t ;], then applies rotation, then moves
again for a fixed time [; \Delta t ;]. This is an approximation to the curve along which the robot
is actually moving, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/19-monte-carlo-localisation-2.png" />

## Localisation Question 1 + 2 ##

Nothing to note.
