Machine Learning Notes
======================

**NOTE:-** I am no longer studying the machine learning course in order to focus on AI, so these
  notes are incomplete. I might come back and attempt to go over the course again later (if that is
  permitted) at which time I will fix 'em :-)

1. Introduction
---------------

## Video: What Is Machine Learning? ##

Arthur Samuel (1959) - "Field of study that gives computers the ability to learn without being
explicitly programmed"

Tom Mitchell (1998) - "Well-posed learning problem: A computer program is said to *learn* from
experience E with respect to some task T and some performance measure P, if its performance on T, as
measured by P, improves with experience E."

Machine learning algorithms are usually of one of the following types:-

* Supervised learning
* Unsupervised learning

(There are others, such as reinforcement learning and recommender systems, but not as popular).

## Video: Supervised Learning ##

E.g. housing price prediction

       ^
       |  x  x
    $k |   x
       | xx
       |
       ---------->
        size ft^2

Can fit a straight line, or perhaps another fit (e.g. quadratic) as appropriate depending on the
learning algorithm.

This is an example of supervised learning. We gave the model 'right answers' i.e. the correct
housing prices, and we effectively want the learning algorithm to give more correct answers.

More formally:- In supervised learning, the algorithm is given data in which it is told the "correct
answer" for each example.

Also known as a regression problem - we're trying to predict a continuous value attribute (though of
course with price you could consider values to be discrete, we assume here that they are, in fact,
continuous).

E.g. breast cancer (malignant vs benign)

        ^
      1 |       x x x x x
    m?  |
        |
        |
      0 *-x-x-x-x--x---->
           tumour size
              ^
              |
      friend's tumour size

Where m? = malignant?, 0 = no, 1 = yes

This is a classification problem as we are trying to predict a value based on a discrete-valued
output that has a small range of possible values (0, 1 in this case).

If tumour size is the "attribute", or "feature" we are going to use to predict malignancy, we can
look at this a different way:-

    -o-o-o-o-o-x-o-x-x-x-x-->
          tumour size

Where o = benign, x = malignant.

One 'feature' or 'attribute' here - tumour size. We may have more, e.g. both age + tumour size:-

        ^
        | o o xo x
    age |  oxo x o
        |   o xo
        |    x   o
        *--------------->
           tumour size

There are other features we could look at, such as clump thickness, uniformity of cell size,
uniformity of cell shape + many others.

In fact, one of the most interesting learning algorithms allows the use of an *infinite* number of
features. How is this possible on a real-world limited resource computer? The 'support vector
machine' uses a neat mathematical trick to permit this.

## Video: Unsupervised Learning ##

In the case of supervised learning, we tell the algorithm what is the so-called 'correct' answer,
e.g. whether breast cancer is malignant or benign.

E.g.:-

       ^
       |           o
       |          o o
    x2 |  x      o
       | x  x
       |  x
       --------------->
              x1

Here the points are labelled, e.g. malignant/benign.

In unsupervised learning, we don't have this luxury + essentially have 'unlabelled' points, e.g.:-

       ^
       |           o
       |          o o
    x2 |  o      o
       | o  o
       |  o
       --------------->
              x1

Essentially - 'here is a dataset, can you find some structure in the data'.

We might decide that different subsets of the data are 'clustered' together, and thus use a
clustering algorithm to determine which points belong to which cluster (e.g. the graph above
clearly exhibits two different clusters).

Some *clustering* e.g.s:-

* google news - http://news.google.com - stories that are about the same subject get clustered
together.

* genetics -clustered genetically similar people together:-

          |
          |
    Genes |  [lots of lovely clustered colours :)]
          |
          |
          *---------------------------------------
                      Individuals

* Determining which machines together more frequently than they communicate with other machines -
  can use this to restructure a data centre to be more efficient.

* Social network analysis - Determine which friends connect with one another the most.

* Market segmentation - Companies often have huge customer databases. Can use this data to classify
  customers into different market segments so you can more efficiently sell to groups of
  customers. Again we don't know in advance who is in a given market segment.

* Astronomical data analysis - A surprising application - useful in determining how galaxies formed,
  for example.

These are all examples of applications of clustering algorithms. Let's consider another approach.

### Cocktail Party Problem ###

People at a cocktail party, all talking with each other. Imagine we place microphones at 2 different
places. Consider 2 speakers, speaker #1 and speaker #2 - #1 might be a bit louder in microphone #1,
and a bit softer in microphone #2, and vice-versa for speaker #2, due to the positions relative to
the two speakers.

There is an algorithm, called the cocktail party algorithm, to which you can give these two
recordings and effectively 'ask it to find structure'. The algorithm is then capable of separating
the conversations.

You might expect this to involve a crazy amount of code. In fact, you can represent it using a
single line of code (octave):-

    [W,s,v] = svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');

E.g. svd = Singular Value Decomposition.

2. Linear Regression With One Variable
--------------------------------------

## Video: Model Representation ##

Taking a look at housing prices once again, e.g.:-

Housing Prices (Portland, OR)

            ^        /
            |      x/ x x
    price   |    x / x x
     $k     |   x /--x x
            |  x /x
            |   /xx
            *-------------------->
                size (feet^2)

this is an example of supervised learning - we're given the "right answer" for each example in the
data.

this is a regression problem, as we are looking at a continuous output.

this is a training set - the "correct" answers.

### training set of housing prices (portland, or) ###


    +--------------------+--------------------+
    | size in feet^2(x)  |px ($) in 1000's (y)|
    +--------------------+--------------------+
    |        2104        |        460         |
    +--------------------+--------------------+
    |        1416        |        232         |
    +--------------------+--------------------+
    |        1534        |        315         |
    +--------------------+--------------------+
    |        852         |        178         |
    +--------------------+--------------------+
    |        ...         |        ...         |
    +--------------------+--------------------+

notation:-

  m            = number of training examples
x's            = "input" variable / features
y's            = "output" variable / "target" variable
(x, y)         - one training example
(x^(i), y^(i)) - ith training example

here we say m = 47.

e.g.
x^(1) = 2104
x^(2) = 1416
y^(1) = 460

the process is as follows:-

training set -> learning algorithm -> h (hypothesis)

the hypothesis takes the input, x, size of house and outputs an output, y, estimated price.

h maps from x's to y's.

perhaps 'hypothesis' is not the best term to use for this, however it has stuck around from the
early days of machine learning.

### how do we represent h? ###

    [; h_{\theta}(x) = \theta_0 + \theta_1x ;]

    shorthand: h(x)

    ^      /x  [; h(x) = \theta_0 + \theta_1 x ];
    |    x/ x
    |   x/ x
    |  x/x
    |  /
    *-/----------->

this is linear regression with one variable / univariate linear regression.

## video: cost function ##

enables us to fit the best possible line to our data.

    Hypothesis: [; h_\theta(x) = \theta_0 + \theta_1x ;]

    [; \theta_i ;]'s: parameters

how to choose the 2 thetas?

different values give us different plots, clearly.

    Idea: choose [; \theta_0 ;], [; \theta_1 ;] so that [; h_\theta(x) ;] is close to y for our
     training examples [; (x, y) ;].

more formally:-

    minimise [; \theta_0 \theta_1 ;]

    i.e. minimise [; \theta_0 ;] and [; \theta_1 ;].

    minimise [; \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)} - y^{(i)})^2) ;]

    where

    [; h_\theta(x^{(i)}) = \theta_0 + \theta_1 x^{(i)} ;]

note that m is the # of training examples.

we choose 1/2m for mathematical convenience.

we define a cost function:-

    [; J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2 ;]

thus our task is to:-

    Minimise [; j(\theta_0, \theta_1) ;]

this cost function is referred to as the 'squared error function'.

squared error function is often the best choice for linear regression problems.

## video: cost function - intuition 1 ##

Hypothesis:-

    [; h_\theta(x) = \theta_0 + \theta_1x ;]

Parameters:-

    [; \theta_0, \theta_1 ;]

Cost function:-

    [; j(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 ;]

Goal:-

    Minimise [; j(\theta_0, \theta_1) ;]

Let's look at a simplified hypothesis:-

    [; h_\theta(x) = \theta_1x ;] (e.g. [; \theta_0 = 0 ;])

Here:-

    [; J(\theta_1) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^m(\theta_1x^{(i)} - y^{(i)})^2 ;]

    minimise [; j(\theta_1) ;]

i.e. passes through the origin.

compare the two functions, i.e. the hypothesis function, h, and the cost function, j:-

     [; h_\theta(x) ;]                           | [; j(\theta_1) ;]
     (for fixed theta_1 this is a function of x) | (function of the parameter theta_1)
                                                 |
       ^                                         |            ^
       |               / theta_1=1               |            |
    3  |             x                           |          3 |
       |           / |                           |            x
    2  |         x   |                           |   j(t_1) 2 |
       |       / |   /- theta_1=0.5              |            |
    1  |     x   /---                            |          1 |
       |   / /---                                |            |
       | /---                                    |            |  x     x
    0  *------------------->                     |          0 *-----x------------->
       0     1   2   3                           |            0     1   2   3 theta_1

    j(theta_1) = 1/(2m) * sigma{i=1->m}((h_theta(x^(i)) - y^(i))^2)
               = 1/(2m) * sigma{i=1->m}((theta_1 * x^(i) - y^(i))^2) = 0

    h_theta(x^(i)) = y^(i)

    so j(1) = 0.

since the training set is literally on the hypothesis line.

    j(0.5) = 1/(2*3) * ((0.5 * 1  - 1)^2 + (0.5 * 2 - 2)^2 + (0.5 * 3 - 3)^2)
           = 1/6     * ((-0.5)^2 + (-1)^2 + (-1.5)^2)
           = 1/6     * (0.25 + 1 + 2.25)
           = 1/6 * 3.5
           = 7/12 =~ 0.583 =~~ 0.6

    j(0)   = 1/(2*3) * ((0 * 1 - 1)^2 + (0 * 2 - 2)^2 + (0 * 3 - 3)^2)
           = 1/6     * (1 + 4 + 9)
           = 14/6 = 7/3 =~ 2.333 =~~ 2.3

    j(-0.5) =~~ 5.15

    j(1.5) = 1/6 * ((1.5 * 1 - 1)^2 + (1.5 * 2 - 2)^2 + (1.5 * 3 - 3)^2)
           = 1/6 * (0.5^2 + 1^2 + 1.5^2) = 1/6 * (0.25 + 1 + 2.25) = 1/6 * 3.5
           =~ 0.583 =~~ 0.6

if we keep on mapping out values, we end up with a parabola (well, at least parabola-like) curve.

for each point on the j(theta_1) curve, there is a corresponding hypothesis curve.

since the objective is to minimise:-

    j(theta_1)

clearly the best choice is:-

    theta_1 = 1

## video: cost function - intuition 2 ##

look at our problem statement again:-

hypothesis:-

    h_theta(x) = theta_0 + theta_1 * x

parameters:-

    theta_0, theta_1

cost function:-

    j(theta_0, theta_1) = 1/(2m) sigma{i=1->m}(h_theta(x^(i))-y^(i))^2

goal:-

    minimise j(theta_0, theta_1)

we are going to use contour plots to gain understanding.

                 h_theta (fixed theta_0, theta_1)     |   j(theta_0, theta_1)
                                                      |
             ^                                        |  ^
             |         x  x x x                       |  |
             |            x                           |  |     /
        px   |          x x x             h_theta(x)  |  |    / plotting 3 values - j, theta_0, theta_1
        $000 |        x   x          /----            |  |   /  could be in 3d!
             |     x x  x    /-------                 |  |  /
             |   x  x/-------    theta_0 = 50         |  | /
             /------- x          theta_1 = 0.66       |  |/
         ----|                                        |  *-------------------->
             *------------------->                    |
                   size/feet^2 (x)                    |  we can, instead, use the aforementioned contour plot,
                                                      |  where we plot a contour plot:-
    shown line -> h_theta(x) = 50 + 0.66x             |
                                                      |     ^
                                                      |     |\
                                                      |     | \
                                                      |     |  \   etc. - concentric ellipses
                                                      | t_1 |\  \  (see vid @ 03:46 :-)
                                                      |     | \  \
                                                      |     |  \  \
                                                      |     |   \  \
                                                      |     *-------------------->
                                                      |               t_0
                                                      |
                                                      | each point represents values of theta_0 and theta_1
                                                      | where j remains the same.
                                                      |
                                                      | j is increasing from inside ellipse -> outwards.
                                                      |

## video: gradient descent ##

'gradient descent' is an algorithm for minimising cost j. it is used throughout machine learning,
also for minimising other functions not just a cost function.

setup:-

    have some function j(theta_0, theta_1)

    want min{theta_0, theta_1}( j(theta_0, theta_1))

    outline:-

    * start with some theta_0, theta_1 - e.g. theta_0 = 0, theta_1 = 0
    * keep on changing theta_0, theta_1 to reduce j(theta_0, theta_1) until we hopefully end up at a
      minimum.

as mentioned, can apply to a more general function, e.g.:-

    j(theta_0, theta_1, theta_2, ..., theta_n)
    min j(theta_0, ..., theta_n)

see vid at 2:15 for bumpy 3d plot :-)

'if i was to take a baby-step in some direction, which direction has the steepest downwards slope?'

end up at a *local* minimum. these can be different minima.

### gradient descent algorithm ###

    repeat until convergence {
        theta_j := theta_j - alpha * d/d(theta_j) j(theta_0, theta_1) for j=0 and j=1
    }

    where d/d(theta_j) is a partial differential.

    ----

    we want to simultaneously update theta_0 and theta_1.

    correct: simultaneous update

    temp0 := theta_0 - alpha * (d/dtheta_0) j(theta_0, theta_1)
    temp1 := theta_1 - alpha * (d/dtheta_1) j(theta_0, theta_1)
    theta_0 := temp0
    theta_1 := temp1

    incorrect: non-simultaneous update

    temp0 := theta_0 - alpha * (d/dtheta_0) j(theta_0, theta_1)
    theta_0 := temp0
    temp1 := theta_1 - alpha * (d/dtheta_1) j(theta_0, theta_1)
    theta_1 := temp1

in addition:-

    note that := denotes assignment.

    e.g. a := b means set a to the value of b.

    contrariwise, a = b is a truth assertion.

    alpha = learning rate, i.e. the size of the 'steps' we're taking.

## video: gradient descent intuition ##

    gradient descent algorithm

    repeat until convergence {
        theta_j := theta_j - alpha * (d/d theta_j) j(theta_0, theta_1)
    }

* alpha is the 'learning rate' and controls how big a step we take when updating theta_j.

* the partial-derivative portion of the function is called the 'derivative'.

let's examine the following problem:-

    attempt to minimise j(theta_1) where theta_1 is a real number.

see video for graph :)

    theta_1 = theta_1 - alpha * (d/d theta_1) j(theta_1)

we look at the *tangent* at a point on the curve.

we can increment *or* decrement the parameter depending on whether the tangent slope is positive or
negative.

if alpha is too small, gradient descent can be slow.

if alpha is too large, gradient descent can overshoot the minimum. it may fail to converge, or even
diverge.

if we are already at the local minimum, then taking another step will not move us away, as the gradient will be 0.

gradient descent can converge to a local minimum, even with the learning rate, alpha, fixed.

as we approach a local minimum, gradient descent will automatically take smaller steps. so, no need
to decrease alpha over time. the gradient varies.

## video: gradient descent for linear regression ##

let's compare our gradient descent algorithm with our linear regression model:-

    gradient descent algorithm

    repeat until convergence {
        theta_j = theta_j - alpha * (d/d theta_j) j(theta_0, theta_1)

        (for j=1, j=0)

    ----

    linear regression model

    h_theta(x) = theta_0 + theta_1 * x

    j(theta_0, theta_1) = 1/(2m) * sigma{i=1->m}(h_theta(x^(i)) - y^(i))^2

    minimise j(theta_0, theta_1).

let's apply gradient descent to minimise our cost function.

we must determine what the partial derivative term evaluates to.

    d/dtheta_j j(theta_0, theta_1) = d/dtheta_j ( 1/(2m) sigma{i=1->m}(h_theta(x^(i)) - y^(i))^2 )

    = d/dtheta_j 1/(2m) sigma{i=1->m}(theta_0 + theta_1 x^(i) - y^(i))^2

we need to determine this term for j=0, and j=1.

the derivation is not shown, but the result is:-

    j=0, d/d(theta_0) j(theta_0, theta_1) = (1/m) * sigma{i=1->m} (h_theta(x^(i)) - y^(i))
    j=1, d/d(theta_1) j(theta_0, theta_1) = (1/m) * sigma{i=1->m} ( (h_theta(x^(i)) - y^(i)) * x^(i) )

(this uses multivariate calculus.)

plugging these values back in to the gradient descent algorithm:-

    repeat until convergence {
        theta_0 = theta_0 - alpha * (1/m) * sigma{i=1->m} (h_theta(x^(i)) - y^(i))
        theta_1 = theta_1 - alpha * (1/m) * sigma{i=1->m} ( (h_theta(x^(i)) - y^(i)) * x^(i))
    }

    these should be updated simultaneously.

we don't need to be concerned about local minima here, as the linear regression cost function is
always a 'bowl' shaped curve (see vid @ 05:00), technical term = "convex function". in this
function, local optimum = global optimum.

let's try this is reality.

let's initialise the calculation at:-

    theta_0 = 900, theta_1 = -0.1

as we continue we eventually find the optimum. woo!

this approach is sometimes called "batch" gradient descent. batch - each step of gradient descent
uses *all* the training examples, i.e. in the summations.

there exists a method for numerically solving for the minimum of the cost function j, without
needing to use an iterative algorithm like gradient descent, however we'll talk about that
later. (this method is called the normal equations method.)

it turns out that gradient descent scales better to larger datasets than the normal equations
method.

## video: what's next ##

two extensions:-

1 extension 1:-

    in min j(theta_0, theta_1) can solve for theta_0, theta_1 exactly without needing iterative
    algorithm (gradient descent).

2 learn with larger number of features.

    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |   size (feet^2)    |  no. of bedrooms   |   no. of floors    | age of home (yrs)  |   price ($1000)    |
    |        x_1         |        x_2         |        x_3         |        x_4         |                    |
    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |        2104        |         5          |         1          |         45         |        460         |
    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |        1416        |         3          |         2          |         40         |        232         |
    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |        1534        |         3          |         2          |         30         |        315         |
    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |        852         |         2          |         1          |         36         |        178         |
    +--------------------+--------------------+--------------------+--------------------+--------------------+


hard to visualise graphically, notation gets more complicated.

turns out linear algebra is very useful for working with this amount of data.

linear algebra very useful for implementing efficient solutions to problems.

3. linear algebra review
------------------------

## video: matrices and vectors ##

what are matrices, what are vectors?

matrix: rectangular array of numbers, e.g.:-

    [ 1402 191  ]    [ 1 2 3 ]
    [ 1371 821  ] or [ 4 5 6 ]
    [ 949  1437 ]
    [ 147  1448 ]

dimension of matrix = rows x cols, e.g. the above are 4x2 and 2x3.

sometimes written as |r^(4x2) and |r^(2x3) to refer to all matrices of the specified dimensions.

we refer to elements of a matrix, a, as a_ij, where i = element row, j = element column.

so in:-

    [ 1402 191  ]
    [ 1371 821  ]
    [ 949  1437 ]
    [ 147  1448 ]

    a_11 = 1402
    a_12 = 191
    a_32 = 1437
    a_41 = 147
    a_43 = undefined (error)

vector is a special case of a matrix - an nx1 matrix.

e.g.:-

        [ 460 ]
        [ 232 ]
    y = [ 315 ]
        [ 178 ] <- 4-dimensional vector.


    |r^4 = set of all 4-dimensional vectors.

    y_i = ith element

    so here y_1 = 460, y_2 = 232, y_3 = 315, y_4 = 178.

there are two different means of indexing into a vector, 1-indexed and 0-indexed. 1-indexed vectors
tend to be more common in mathematics, and this is how we will refer to vector elements most of the
time on the course, however sometimes useful in machine learning to use 0-indexed.

usually use upper-case to refer to matrices, e.g. a, b, c, x, lower-case to refer to vectors,
e.g. a, b, x, y.

## video: addition and scalar multiplication ##

how do we add matrices? add up the elements, e.g.:-

    [ 1 0 ]   [ 4 0.5 ]   [ 5 0.5 ]
    [ 2 5 ] + [ 2 5   ] = [ 4 10  ]
    [ 3 1 ]   [ 0 1   ]   [ 3 2   ]

can only add matrices of the same dimensions (here 3x2).

to multiply by a scalar, multiply each element by the scalar.

    3*[ 1 0 ]   [ 3 0  ]
      [ 2 5 ] = [ 6 15 ]
      [ 3 1 ]   [ 9 3  ]

scalar multiplication is commutative.

we can also divide:-

    [ 4 0 ]              [ 4 0 ]    [ 1   0   ]
    [ 6 3 ] / 4 =  1/4 * [ 6 3 ] =  [ 3/2 3/4 ]

can combine the operations:-

      [1]   [0]   [3]
      [4]   [0]   [0]/3
    3*[2] + [5] - [2]

      [ 3  ]   [0]   [1  ]
      [ 12 ]   [0]   [0  ]
    = [ 6  ] + [5] - [2/3]

      [ 2      ]
      [ 12     ]
    = [ 10 1/3 ]

## video: matrix vector multiplication ##

e.g.:-

    [ 1 3 ]         [ 1*1 + 3*5 ] = [ 16 ]
    [ 4 0 ] [ 1 ]   [ 4*1 + 0*5 ] = [ 4  ]
    [ 2 1 ] [ 5 ] = [ 2*1 + 1*5 ] = [ 7  ]

dimensions: 3x2 x 2x1 = 3x1

       a   * x  =  y

    [      ][ ]   [ ]
    [      ][ ] = [ ]
    [      ][ ]

     m x n  nx1 = m-dimensional vector

to get y_i, multiply a's ith rows with elements of vector x, and add them up.

so,

    y_i = sigma[j=1->m](a_ij * x_j)

e.g.:-

                   [ 1 ]
    [ 1  2  1  5 ] [ 3 ]   [ 1*1  + 2*3  + 1*2 + 5*1 ]   [ 1  + 6  + 2 + 5 ]   [ 14 ]
    [ 0  3  0  4 ] [ 2 ] = [ 0*1  + 3*3  + 0*2 + 4*1 ] = [ 0  + 9  + 0 + 4 ] = [ 13 ]
    [ -1 -2 0  0 ] [ 1 ]   [ -1*1 + -2*3 + 0*2 + 0*1 ]   [ -1 + -6 + 0 + 0 ]   [ -7 ]

e.g. :-

    house sizes:

    2104
    1416
    1534
    852

    h_theta(x) = -40 + 0.25x

can construct as matrix/vector multiplication:-

      4x2          2x1    =  4x1

    [ 1 2104 ]   [ -40  ]   [ 1*-40 + 2104*0.25 ]   [ h_theta(2104) ]   [ 486   ]
    [ 1 1416 ] x [ 0.25 ] = [ 1*-40 + 1416*0.25 ] = [ h_theta(1416) ] = [ 314   ]
    [ 1 1534 ]              [ 1*-40 + 1434*0.25 ]   [ h_theta(1434) ]   [ 318.5 ]
    [ 1 852  ]              [ 1*-40 + 852*0.25  ]   [ h_theta(852)  ]   [ 173   ]

    e.g. prediction = data matrix * parameters

## video: matrix matrix multiplication ##

e.g.:-
             [ 1 3 ]
    [ 1 3 2 ][ 0 1 ]
    [ 4 0 1 ][ 5 2 ]

treat like matrix/vector multiplication:-

             [ 1 ]
    [ 1 3 2 ][ 0 ] = [ 11 ]
    [ 4 0 1 ][ 5 ]   [ 9  ]

            [ 3 ]
    [ 1 3 2][ 1 ] = [ 10 ]
    [ 4 0 1][ 2 ]   [ 14 ]

then put these together:-

             [ 1 3 ]
    [ 1 3 2 ][ 0 1 ] = [ 11 10 ]
    [ 4 0 1 ][ 5 2 ]   [ 9  14 ]

       2x3     3x2   =    2x2

    a (m x n) x b (n x o) = c (m x o)

the ith column of the matrix c is obtained by multiplying a with the ith column of b (for i = 1, 2,
..., o).

e.g.:-

    [ 1 3 ] [ 0 1 ]   [ 9  7  ]
    [ 2 5 ] [ 3 2 ] = [ 15 12 ]

another 'neat trick', e.g.:-

   house sizes:

    2104
    1416
    1534
    852

only this time we have 3 competing hypotheses:-

    1. h_theta(x) = -40  + 0.25x
    2. h_theta(x) = 200  + 0.1x
    3. h_theta(x) = -150 + 0.4x

can express this as a matrix multiplication problem:-

    matrix            matrix

    [ 1 2104 ]   [ -40  200 -150 ]   [ 486   200.1 691.6 ]
    [ 1 1416 ] x [ 0.25 0.1 0.4  ] = [ 314   341.6 416.4 ]
    [ 1 1534 ]                       [ 343.5 353.4 463.6 ]
    [ 1 852  ]                       [ 173   285.2 190.8 ]

each column in resultant matrix is predicted house prices for respective hypothesis.

good linear algebra libraries out there to use

## video: matrix multiplication properties ##

w.r.t scalars, matrix multiplication is commutative (i.e. order of operands can change).

let a and b be matrices. then in general, a x b != b x a (not commutative.)

e.g.:-

    [ 1 1 ] [ 0 0 ] = [ 2 0 ]
    [ 0 0 ] [ 2 0 ]   [ 0 0 ]

    [ 0 0 ] [ 1 1 ] = [ 0 0 ]
    [ 2 0 ] [ 0 0 ] = [ 2 2 ]

also, consider multiplying an mxn matrix, a, by and nxm matrix b:-

    mxn x nxm -> mxm
    nxm x mxn -> nxn

so in this case, i.e. one where it is possible to swap the order of matrices, the two matrices even
have different dimensions.

e.g. in the case of scalars:-

    3 x 5 x 2 can be represented as:-

    3  x 10 or
    15 x 2

so:-

    3x(5x2) = (3x5)x2

scalar multiplication is "associative", i.e. a(bc) = (ab)c.

this is true of matrices. so we can solve a x b x c thusly:-

let d = b x c. compute a x d.
let e = a x b. compute e x c.

### identity matrix ###

in scalar numbers, 1 is identity. so 1 x z = z x 1 = z, for any z.

denoted i, or i_(nxn).

examples of identity matrices:-

    [ 1 0 ]
    [ 0 1 ]     2x2

    [ 1 0 0 ]
    [ 0 1 0 ]
    [ 0 0 1 ]   3x3

    [ 1 0 0 0 ]
    [ 0 1 0 0 ]
    [ 0 0 1 0 ]
    [ 0 0 0 1 ] 4x4

1's along the diagonal, 0's everywhere else.

informally:-

(zeroes are big)

[ 1
   1   0
    1
     .
      .
   0   .
        1 ]

a   . i   = i   . a   = a

mxn x nxn = mxm x mxn = mxn

so the dimensions of i have to vary here.

note, ab != ba in general, but if b = i, then ai = ia.

## video: inverse and transpose ##

as mentioned previously, in real numbers, 1 = "identity".

each real number has an inverse, such that ab = i. so e.g. 3, inverse = 3^-1 = 1/3.

not all numbers have an inverse, i.e. 0 - 0^-1 is undefined.

### matrix inverse ###

if a is an m x m matrix, and if it has an inverse, then:-

    a(a^-1) = (a^-1)a = i.

note mxm matrix - 'square matrix', as rows = cols. only square matrices have inverses.

e.g.

           [ 3 4  ]
    a =    [ 2 16 ] 2x2 so inversable.

    a^1 = [ 0.4   -0.1  ]
          [ -0.05 0.075 ]

    [ 3 4  ] [ 0.4   -0.1  ]   [ 1 0 ]
    [ 2 16 ] [ -0.05 0.075 ] = [ 0 1 ] = i_(2x2)

       a           a^-1

can compute easily in octave:-

    octave-3.2.3:1> a = [ 3 4; 2 16]
    a =

        3    4
        2   16

    octave-3.2.3:2> pinv(a)
    ans =

       0.400000  -0.100000
      -0.050000   0.075000

    octave-3.2.3:3> inverseofa = pinv(a)
    inverseofa =

       0.400000  -0.100000
      -0.050000   0.075000

    octave-3.2.3:4> a*inverseofa
    ans =

       1.00000   0.00000
      -0.00000   1.00000

    octave-3.2.3:5> inverseofa*a
    ans =

       1.0000e+00  -4.4409e-16
       2.7756e-17   1.0000e+00

some matrices don't have inverses, e.g.:-

        [ 0 0 ]
    a = [ 0 0 ]

intuition: don't have an inverse if 'too close to zero'

matrices that don't have an inverse are "singular" or "degenerate"

### matrix transpose ###

e.g.:-

          [ 1 2 0 ]
    a =   [ 3 5 9 ] 2x3

    a^t = [ 1 3 ] = b
          [ 2 5 ]
          [ 0 9 ]  3x2

flipping along a 45 degree axis. or swapping rows + columns!

let a be an mxn matrix, and let b = a^t.

then b is an nxm matrix, and bij = aji.

4. linear regression with multiple variables
--------------------------------------------

## Multiple Features ##

Let's say we had more features, rather than just size + price.

We denote features:-

    [; x_1, x_2, x_3, x_4 ;]

Also:-

    n = no. of features
    [; x^{(i)} ;] = input (features) of [; i^{th} ;] training example.
    [; x_j^{(i)} ;] = value of feature j in [; i^{th} ;] training example.

    [; X^{(2)} ;] is an N-dimensional vector representing training set 2.

    [; X_3^{(2)};] = 2

Hypothesis:-

Previously:-

    [; h_\theta(x) = \theta_0 + \theta_1x ;]

Now:-

    [; h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3+x_3 + \theta_4x_4 ;]

E.g.:-

    [; h_\theta(x) = 80 + 0.1x_1 + 0.01x_2 + 3x_3 -2x_4 ;]


For n features:-

    [; h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n ;]

For convenience of notation, define [; x_0 = 1 ;].

So:-

    [; (x^{(i)}_0 = 1) ;]

And:-

    x = [x0]   t = [t0]
        [x1]       [t1]
        [x2]       [t2]
        [..]       [..]
        [xn]       [tn]

We can then write our hypothesis as follows:-

    [; h_\theta(x) = \theta_0x_0 + \theta_1x_1 + ... + \theta_nx_n ;]
    [; = \theta^Tx ;]

Which is convenient.

This is called multivariate linear regression, i.e. multivariate = multiple features/variables.

## Gradient Descent for Multiple Variables ##

    Hypothesis: [; h_\theta(x) = \theta^Tx = \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n ;]

    Parameters: [; \theta_0, \theta_1, ..., \theta_n ;] or [; \theta ;] - n+1-dimensional vector

Cost Function:-

    [; J(\theta) = \frac{1}{2m}\Sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2 ;]

Gradient Descent:-

    Repeat {
        [; \theta_j := \theta_j - \alpha partial deriv d/d\theta_j J(\theta_0, ..., \theta_n) ;]
    }

Simultaneously update for every j= 0, ..., n.

Our new gradient descent algorithm:-

    Repeat {

        [; \theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} ;]

    }

Simultaneously update [; \theta_j ;] for j = 0, ..., n.

Different from our previous algorithm.

Note we use [; x_0 ;] here again.

## Gradient Descent in Practice I - Feature Scaling ##

Practical tricks for making gradient descent work well.

Idea: Make sure features on a similar scale.

E.g. x1 = size (0-2000 feet^2)
x2 = number of bedrooms (1-5)

<img src="http://codegrunt.co.uk/images/ml/gradient-descent-in-practice-i-feature-scaling-1.png" />

Can take a long time to find the global minimum, e.g. long, thin contours.

Effectively normalisation, e.g.:-

    [; x_1 = \frac{size(feet^2)}{2000} ;]

    [; x_2 = \frac{no. of bedrooms}{5}

<img src="http://codegrunt.co.uk/images/ml/gradient-descent-in-practice-i-feature-scaling-2.png" />

Where:-

    [; 0 \leq x_1 \leq 1 ;]
    [; 0 \leq x_2 \leq 2 ;]

More generally:-

Get every feature into approximately a [; -1 \leq x_i \leq 1 ;] range.

-1, 1 are relatively arbitrary. Only have to roughly hit it.

The problem comes when there are differences in order of magnitude.

Good:-

    [; 0 \leq x_1 \leq 3 ;]
    [; -2 \leq x_2 \leq 0.5 ;]

Bad:-

    [; -100 \leq x_3 \leq 100 ;]
    [; -0.0001 \leq x_4 \leq 0.0001 ;]

E.g. -3 to 3 is good enough.

### Mean Normalisation ###

Replace

    [; x_i ;]

with

    [; x_i - \mu_i ;]

To make features have approx. zero mean (don't apply to [; x_0 = 1 ;]).

E.g.:-

If av. size = 1000

    [; x_1 = \frac{size-1000}{2000} ;]

If av. size is 1-5 bedrooms, around 2:-

    [; x_2 = \frac{#bedrooms-2}{5} ;]

    [; -0.5 \leq \ x_1 \leq 0.5, -0.5 \leq x_2 \leq 0.5 ;]


In general:-

    [; x_1 <- \frac{x_1-\mu_1}{s_1} ;]

Where [; \mu_1 ;] is the average value of x in the training set

And [; S_1 ;] is the range (max-min) or standdev.

This all helps convergence happen a lot quicker.

## Gradient Descent in Practice II - Learning Rate ##

Gradient descent

    [; \theta_j := \theta_j - \alpha partial d/d\theta_jJ(\theta)

* "Debugging": How to make sure gradient descent is working correctly.

* How to choose learning rate [; \alpha ;].

<img src="http://codegrunt.co.uk/images/ml/gradient-descent-in-practice-ii-learning-rate-1.png" />

If the gradient descent is working correctly, then [; J(\theta) ;] should decrease after every iteration.

No. of steps to converge can vary a *lot*.

Very hard to tell in advance how many iterations it will take.

It's actually possible to write an automatic convergence test - e.g. declare convergence if
[; J(\theta) ;] decreases by less than [; 10^{-3} ;] in one iteration.

Choosing the threshold is difficult. Often better to just look at a plot.

If [; J(\theta) ;] is actually increasing, then usually indicative of too large [; \alpha ;] and
overshooting the minimum on a parabolic cost function curve.

* For sufficiently small [; \alpha ;], [; J(\theta) ;] should decrease on every iteration.
* But if [; \alpha ;] is too small, gradient descent can be slow to converge.

* If [; \alpha ;] is too small: slow convergence.
* If [; \alpha ;] is too large: [; J(\theta) ;] may not decrease on every iteration; may not converge.

To choose [; \alpha ;], try:-

    0.001, 0.01, 0.1, 1, ...

Plot [; J(\theta) ;] vs. # iterations for each.

Andrew actually tends to jump like this:-

    0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, ...

i.e. x3 each time.

## Features and Polynomial Regression ##

Choice of features - how to get powerful learning algorithms going.

How to use polynomials, using the machinery of linear regression, to fit more complicated hypotheses.

E.g. housing prices prediction:-

    [; h_\theta(x) = \theta_0 + \theta_1 \times frontage + \theta_2 \times depth ;]

I might use area instead, e.g.:-

    [; x = frontage \times depth ;]

    [; h_\theta(x) = \theta_0 + \theta_1 x ;]

So can use insight to choose a better model.

E.g. we might find a quadratic model is a better fit for some data, e.g.:-

    [; \theta_0 + \theta_1x + \theta_2x^2 ;]

Or perhaps using a cubic function:-

    [; \theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 ;]

In order to use an approach like this, then we can simply replace

    [; x_1, x_2, x_3 ;]

With:-

    [; x_1 = (size) ;]
    [; x_2 = (size)^2 ;]
    [; x_3 = (size)^3 ;]

So:-

    [; h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 ;]
    [; = \theta_0 + \theta_1(size) + \theta_2(size)^2 + \theta_3(size)^3 ;]

If you choose your features like this, then feature scaling becomes considerably more important, e.g.:-

size:           1 - 1,000
size^2:         1 - 1,000,000
size^3:         1 - 10^9

Can choose arbitrary functions, e.g.:-

    h_\theta(x) = \theta_0 + \theta_1(size) + \theta_2\sqrt(size)

Again, feature scaling matters.

Later we will look at algorithms which will make function fitting choices automatically.

## Normal Equation ##

The algorithm we've been using so far is gradient descent - making multiple iterations of gradient descent to converge to the minimum

Normal equation: Method to solve for [; \theta ;] analytically.

Intuition: If 1D ([; \theta ;] is a real number)

    [; J(\theta) = a\theta^2 + b\theta + c ;]

We can solve for the minimum of this curve using:-

    [; d/d\thetaJ(\theta) = ... set 0. ;]

Solve for [; \theta ;].

    [; J(\theta_0, \theta_1, ..., \theta_m) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 ;]

We can use partial derivatives, for all values of j:-

    [; partial d/d\theta_j = ... = 0 ;]

Solve for:-

    [; \theta_0, \theta_1, ... , \theta_n ;]

E.g. m = 4 training examples.

Then we can construct a matrix, X, which contains all the training data, including [; x_0 ;].

X is m x n+1 matrix, y is an m-dimensional vector.

We can now minimise the cost function using the following function:-

    [; \theta = (X^T X)^{-1}X^Ty ;]

Let's consider that we have m examples ([; x^{(1)}, y^{(1)}), ..., (x^{(m)},y^{(m)}) ;], n features.

E.g., if [; x^{(i)} ;] =

    [1]
    [x_1^{(i)}]

Then:-
        [ 1 X_1^{(1)} ]
    X = [ 1 X_2^{(1)} ]
        [     .       ]
        [     .       ]
        [ 1 X_m^{(1)} ]

Which is an mx2 matrix.

    y = [ y^{(1)} ]
        [ y^{(2)} ]
        [    .    ]
        [    .    ]
        [ y^{(m)} ]

Looking at the equation again:-

    [; \theta = (X^TX)^{-1}X^Ty ;]

    [; (X^TX)^{-1} ;]

Is inverse of matrix

    [; X^TX ;]

In Octave:-

    pinv(X'*X)*X'*y

You represent transposes in Octave via:-

    X'

If you use the normal equation, then feature scaling isn't such a big deal.

When should you use gradient descent vs. normal equation?

### Gradient Descent ###

* Need to choose [; \alpha ;].
* Needs many iterations.
* Works well, even when n is large.

### Normal Equation ###

* No need to choose [; \alpha ;].
* Don't need to iterate.
* Needs to compute [; (X^TX)^{-1} ;], which is nxn. Inverse is [; O(n^3) ;]
* Slow if n is very large.
* If n = 100, no problem.
* If n = 1000, still ok.
* If n = 10,000 then we're getting cold feet.
* If n = 10^6, then we're bailing.

Normal equation method is great alternative to gradient descent when n is small, e.g. 1000.

Normal equation not so great for more sophisticated learning algorithms.

## Normal Equation Noninvertibility ##

(optional, skip for now)

5. Octave Tutorial
------------------

Basic Operations
----------------

Octave is the best choice for investigating machine learning since it's so high-level. Can prototype
in octave, then re-implement in other language.

Octave vs. matlab vs. R vs. numpy - octave tends to be the easiest choice.

Basic Commands:-

Simple maths:-

    5+6
    3-2
    etc.

Comments:-
    % for comments

Logical

    1 == 2 % false
    1 ~= 2 % not equals
    1 && 0 % AND
    1 || 0 % OR
    XOR(1,0)

Can change prompt:-

    PS1('>> ');

Can assign variables:-

    a = 3

If you don't want to print out a result, use the semicolon:-

    a = 3;    % semicolon suppressing output

You can output things either by simply stating the var name, or using disp():-

    disp(a);

    disp(sprintf('2 decimals: %0.2f', a))

    disp(sprintf('6 decimals: %0.6f', a))

Can default to expanding things to long format via:-

    format long

Can assign matrices via:-

    A = [1 2; 3 4; 5 6]

Can also do:-

    A = [1 2;
    3 4;
    5 6]

Similarly for vectors:-

    v = [1 2 3]

Which is a 1x3 vector

Can input a 3x1 vector thusly:-

    v = [1; 2; 3]

Can input ranges:-

    v = 1:0.1:2

Or:-

    v = 1:6

Can set up a matrix with all values = 1, e.g.:-

    ones(2,3)

Can multiply to get other vals:-

    C = 2*ones(2, 3)

also can do for zeros:-

    w = zeros(2, 3)

Can get random variables, e.g.:-

    w = rand(3, 3)

Gives 3x3 matrix with random variables 0-1.

Each time you run it you get a different set of values.

Can get guassian random numbers using randn, e.g.:-

    w = randn(1, 3)

Can look at huge sets:-

    w = -6 + sqrt(10)*(randn(1, 10000))

And draw a histogram:-

    hist(w)

Can plot with more bins/buckets:-

    hist(w, 50)

Can get the identity matrix via the eye() command, e.g.:-

    eye(4)

Can use the help command to get details on commands, etc., e.g.:-

    help eye

