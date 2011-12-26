AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 6 - Unsupervised Learning
------------------------------

## Unsupervised Learning + Question ##

We've talked about supervised learning, in which we have data and target labels. In unsupervised
learning we just have data, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/6-unsupervised-learning-question-1.png" />

Here we have m items of n features each.

The task of unsupervised learning is to find structure in data of this type.

Let's consider the following graph:-

<img src="http://codegrunt.co.uk/images/ai/6-unsupervised-learning-question-2.png" />

This is not entirely random - there is structure, consisting of 2 clusters values.

Looking at the following example:-

<img src="http://codegrunt.co.uk/images/ai/6-unsupervised-learning-question-3.png" />

This is in a 2-dimensional space, however intuitively you can see that it could be represented in
1-dimension, as the points on the graph (to an approximation) fit on a line.

## Terminology ##

If we consider the following data:-

<img src="http://codegrunt.co.uk/images/ai/6-terminology-1.png" />

To investigate this using unsupervised learning, it must be 'IID', i.e. identically distributed and
independently drawn from the same distribution.

A key part of the practice of unsupervised learning is performing the task of determining the
underlying density/probability distribution which generated the data. This is known as 'density
estimation', which encompasses:-

* Clustering,
* Dimensionality reduction.
* Many others.

One of the more interesting types of unsupervised learning is 'blind signal separation'. Consider if
you had a microphone into which two separate people talk and you record the combination of their
voices - blind signal separation is concerned with being able to recover the separate voices via
filtering the data into two separate streams. Very complicated. Can be treated as 'factor analysis',
where each speaker is considered to be a factor in the joint signal the microphone records.

There are many other examples of unsupervised learning, some of which we cover below.

## Google Street View and Clustering ##

Unsolved problem. Consider Google Street View - there is a lot of regularity in the street view
imagery, e.g. cars, trees, buildings, pavement, etc. The problem is concerned with taking hundreds
of billions of images which comprise the street view dataset and deriving from them the concepts of
trees, buildings, pedestrians, etc. - this is an unsupervised learning problem. Attempts to do this
have resulted in very small image sets.

Compare this to the human process - we can often derive knowledge of such things simply by
*observing* without target labels. It is one of the biggest unsolved problems in AI - essentially
teaching the computer to derive concepts this way.

Clustering is the most common form of unsupervised learning. Two of the most popular algorithms for
detecting clustering are:-

* k-Means
* Expectation Maximisation - probabilistic generalisation of k-Means derived from first principles.

## k-Means Clustering Example ##

Consider the follow points in a 2d space:-

<img src="http://codegrunt.co.uk/images/ai/6-k-means-clustering-example-1.png" />

k-Means estimates for a fixed number of k (here k=2) the best centres of clusters representing the data points.

We find these centres by following the k-Means algorithm:-

1. Guess centres at random:-

<img src="http://codegrunt.co.uk/images/ai/6-k-means-clustering-example-2.png" />

2. Assign to each cluster centre the most likely corresponding data points by minimising 'Euclidian distance':-

<img src="http://codegrunt.co.uk/images/ai/6-k-means-clustering-example-3.png" />

Each cluster centre represents half of the space, and the line that separates the two centres
represents the equidistant line. This is often referred to as a 'Voronoi graph'. The points on the
left belong to the red cluster, and the points on the right to the green cluster.

3. Now we have a correspondence between the data points and cluster centres, we move the points such
as to minimise joint quadratic distance from all of each centre's points:-

<img src="http://codegrunt.co.uk/images/ai/6-k-means-clustering-example-4.png" />

4. Iterate. Reassign cluster centres, which gives us a different Voronoi diagram:-

<img src="http://codegrunt.co.uk/images/ai/6-k-means-clustering-example-5.png" />

Now we can re-evaluate cluster centre positions, moving from:-

<img src="http://codegrunt.co.uk/images/ai/6-k-means-clustering-example-6.png" />

To:-

<img src="http://codegrunt.co.uk/images/ai/6-k-means-clustering-example-7.png" />

Since the points no longer change 'hands' between red and green, we have converged.

## k-Means Algorithm + Questions 1-2 ##

Very easy to implement. The algorithm:-

    Initially: select k cluster centres randomly
    
    repeat
        Correspond data points to nearest cluster
        Update cluster centre by means of corresponding data points
        Empty cluster centres: restart at random
    until no change

Known to converge to a local optimum. The general (global) clustering problem is known to be
NP-hard, so a locally optimal solution is really the best we can do with it.

Note you have to take care of the case where cluster centres have *no* data points - in that case,
you have to restart the process by placing cluster centres in new random locations.

Problems with k-means:-

* Need to know k
* Local minima, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/6-k-means-algorithm-questions-1-2-1.png" />

* High dimensionality
* Lack of mathematical basis.

## Expectation Maximisation ##



## Gaussian Learning ##

## Maximum Likelihood ##

## Question 1+2 ##

## Gaussian Summary ##

## EM as Generalisation of k-Means ##

## EM Algorithm ##

## Question 1+2 ##

## Choosing k ##

## Clustering Summary ##

## Dimensionality Reduction ##

## Question ##

## Linear Dimensionality Reduction ##

## Face Example ##

## Scan Example ##

## Piece-Wise Linear Projection ##

## Spectral Clustering ##

## Spectral Clustering Algorithm ##

## Question ##

## Supervised vs. Unsupervised Learning ##
