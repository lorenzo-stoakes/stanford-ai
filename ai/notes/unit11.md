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

    [; S_1 to S_N ;]

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

## Stationary Distribution + Question ##



## Finding Transition Probabilities + Question ##

## Laplacian Smoothing Question ##

