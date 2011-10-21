Machine Learning Notes
======================

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
single line of code (octave/matlab?):-

    [W,s,v] = svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');

E.g. svd = Singular Value Decomposition.

2. Linear Regression With One Variable
--------------------------------------

## Video: Model Representation ##

Taking a look at housing prices once again, e.g.:-

Housing Prices (Portland, OR)

            ^        /
            |      x/ x x
    Price   |    x / x x
     $k     |   x /--x x
            |  x /x
            |   /xx
            *-------------------->
                Size (feet^2)

This is an example of supervised learning - we're given the "right answer" for each example in the
data.

This is a regression problem, as we are looking at a continuous output.

This is a training set - the "correct" answers.

### Training set of housing prices (Portland, OR) ###


    +--------------------+--------------------+
    | Size in feet^2(x)  |Px ($) in 1000's (y)|
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

Notation:-

  m            = Number of training examples
x's            = "input" variable / features
y's            = "output" variable / "target" variable
(x, y)         - one training example
(x^(i), y^(i)) - ith training example

Here we say m = 47.

E.g.
x^(1) = 2104
x^(2) = 1416
y^(1) = 460

The process is as follows:-

Training Set -> Learning Algorithm -> h (hypothesis)

The hypothesis takes the input, x, size of house and outputs an output, y, estimated price.

h maps from x's to y's.

Perhaps 'hypothesis' is not the best term to use for this, however it has stuck around from the
early days of machine learning.

### How do we represent h? ###

    h_theta(x) = theta_0 + theta_1 * x

    Shorthand: h(x)

    ^      /x h(x) = theta_0 + theta_1 x
    |    x/ x
    |   x/ x
    |  x/x
    |  /
    *-/----------->

This is Linear regression with one variable / univariate linear regression.

## Video: Cost Function ##

Enables us to fit the best possible line to our data.

    Hypothesis: h_theta(x) = theta_0 + theta_1 * x

    Theta_i's: parameters

How to choose the 2 thetas?

Different values give us different plots, clearly.

    Idea: choose theta_0, theta_1 so that h_theta(x) is close to y for our training examples (x, y).

More formally:-

    minimise theta_0 theta_1 <- i.e. minimise theta_0 and theta_1 (?)

    Minimise (1/2m) * Sigma[i=1->m](h_theta(x^(i)) - y^(i))^2

    Where h_theta(x^(i)) = theta_0 + theta_1 x^(i)

Note that m is the # of training examples.

We choose 1/2m for mathematical convenience.

We define a cost function:-

    J(theta_0, theta_1) = (1/2m) * Sigma[i=1->m](h_theta(x^(i)) - y^(i))^2

Thus our task is to:-

    Minimise J(theta_0, theta_1)

This cost function is referred to as the 'squared error function'.

Squared error function is often the best choice for linear regression problems.

## Video: Cost Function - Intuition 1 ##

Hypothesis:-

    h_theta(x) = theta_0 + theta_1 * x

Parameters:-

    theta_0, theta_1

Cost Function:-

    J(theta_0, theta_1) = 1/(2m) sigma{i=1->m}(h_theta(x^(i))-y^(i))^2

Goal:-

    Minimise J(theta_0, theta_1)

Let's look at a simplified hypothesis:-

    h_theta(x) = theta_1 * x (e.g. theta_0 = 0)

Here:-

    J(theta_1) = 1/(2m) * Sigma{i=1->m}(h_theta(x^(i)) - y^(i))^2 = 1/(2m) * Sigma{i=1->m}(theta_1 * x^(i) - y^(i))^2

    Minimise J(theta_1)

I.e. passes through the origin.

Compare the two functions, i.e. the hypothesis function, h, and the cost function, J:-

     h_theta(x)                                  | J(theta_1)
     (for fixed theta_1 this is a function of x) | (function of the parameter theta_1)
                                                 |
       ^                                         |            ^
       |               / theta_1=1               |            |
    3  |             x                           |          3 |
       |           / |                           |            x
    2  |         x   |                           |   J(t_1) 2 |
       |       / |   /- theta_1=0.5              |            |
    1  |     x   /---                            |          1 |
       |   / /---                                |            |
       | /---                                    |            |  x     x
    0  *------------------->                     |          0 *-----x------------->
       0     1   2   3                           |            0     1   2   3 theta_1

    J(theta_1) = 1/(2m) * Sigma{i=1->m}((h_theta(x^(i)) - y^(i))^2)
               = 1/(2m) * Sigma{i=1->m}((theta_1 * x^(i) - y^(i))^2) = 0

    h_theta(x^(i)) = y^(i)

    So J(1) = 0.

Since the training set is literally on the hypothesis line.

    J(0.5) = 1/(2*3) * ((0.5 * 1  - 1)^2 + (0.5 * 2 - 2)^2 + (0.5 * 3 - 3)^2)
           = 1/6     * ((-0.5)^2 + (-1)^2 + (-1.5)^2)
           = 1/6     * (0.25 + 1 + 2.25)
           = 1/6 * 3.5
           = 7/12 =~ 0.583 =~~ 0.6

    J(0)   = 1/(2*3) * ((0 * 1 - 1)^2 + (0 * 2 - 2)^2 + (0 * 3 - 3)^2)
           = 1/6     * (1 + 4 + 9)
           = 14/6 = 7/3 =~ 2.333 =~~ 2.3

    J(-0.5) =~~ 5.15

    J(1.5) = 1/6 * ((1.5 * 1 - 1)^2 + (1.5 * 2 - 2)^2 + (1.5 * 3 - 3)^2)
           = 1/6 * (0.5^2 + 1^2 + 1.5^2) = 1/6 * (0.25 + 1 + 2.25) = 1/6 * 3.5
           =~ 0.583 =~~ 0.6

If we keep on mapping out values, we end up with a parabola (well, at least parabola-like) curve.

For each point on the J(theta_1) curve, there is a corresponding hypothesis curve.

Since the objective is to minimise:-

    J(theta_1)

Clearly the best choice is:-

    theta_1 = 1

## Video: Cost Function - Intuition 2 ##

Look at our problem statement again:-

Hypothesis:-

    h_theta(x) = theta_0 + theta_1 * x

Parameters:-

    theta_0, theta_1

Cost Function:-

    J(theta_0, theta_1) = 1/(2m) sigma{i=1->m}(h_theta(x^(i))-y^(i))^2

Goal:-

    Minimise J(theta_0, theta_1)

We are going to use contour plots to gain understanding.

                 h_theta (fixed theta_0, theta_1)     |   J(theta_0, theta_1)
                                                      |
             ^                                        |  ^
             |         x  x x x                       |  |
             |            x                           |  |     /
        Px   |          x x x             h_theta(x)  |  |    / Plotting 3 values - J, theta_0, theta_1
        $000 |        x   x          /----            |  |   /  Could be in 3d!
             |     x x  x    /-------                 |  |  /
             |   x  x/-------    theta_0 = 50         |  | /
             /------- x          theta_1 = 0.66       |  |/
         ----|                                        |  *-------------------->
             *------------------->                    |
                   Size/feet^2 (x)                    |  We can, instead, use the aforementioned contour plot,
                                                      |  where we plot a contour plot:-
    Shown line -> h_theta(x) = 50 + 0.66x             |
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
                                                      | Each point represents values of theta_0 and theta_1
                                                      | where J remains the same.
                                                      |
                                                      | J is increasing from inside ellipse -> outwards.
                                                      |

## Video: Gradient Descent ##

'Gradient descent' is an algorithm for minimising cost J. It is used throughout machine learning,
also for minimising other functions not just a cost function.

Setup:-

    Have some function J(theta_0, theta_1)

    Want min{theta_0, theta_1}( J(theta_0, theta_1))

    Outline:-

    * Start with some theta_0, theta_1 - e.g. theta_0 = 0, theta_1 = 0
    * Keep on changing theta_0, theta_1 to reduce J(theta_0, theta_1) until we hopefully end up at a
      minimum.

As mentioned, can apply to a more general function, e.g.:-

    J(theta_0, theta_1, theta_2, ..., theta_n)
    min J(theta_0, ..., theta_n)

See vid at 2:15 for bumpy 3d plot :-)

'If I was to take a baby-step in some direction, which direction has the steepest downwards slope?'

End up at a *local* minimum. These can be different minima.

### Gradient Descent Algorithm ###

    Repeat until convergence {
        theta_j := theta_j - alpha * d/d(theta_j) J(theta_0, theta_1) for j=0 and j=1
    }

    Where d/d(theta_j) is a partial differential.

    ----

    We want to simultaneously update theta_0 and theta_1.

    Correct: Simultaneous update

    temp0 := theta_0 - alpha * (d/dtheta_0) J(theta_0, theta_1)
    temp1 := theta_1 - alpha * (d/dtheta_1) J(theta_0, theta_1)
    theta_0 := temp0
    theta_1 := temp1

    Incorrect: Non-simultaneous update

    temp0 := theta_0 - alpha * (d/dtheta_0) J(theta_0, theta_1)
    theta_0 := temp0
    temp1 := theta_1 - alpha * (d/dtheta_1) J(theta_0, theta_1)
    theta_1 := temp1

In addition:-

    Note that := denotes assignment.

    E.g. a := b means set a to the value of b.

    Contrariwise, a = b is a truth assertion.

    Alpha = learning rate, i.e. the size of the 'steps' we're taking.

## Video: Gradient Descent Intuition ##

    Gradient descent algorithm

    repeat until convergence {
        theta_j := theta_j - alpha * (d/d theta_j) J(theta_0, theta_1)
    }

* alpha is the 'learning rate' and controls how big a step we take when updating theta_j.

* The partial-derivative portion of the function is called the 'derivative'.

Let's examine the following problem:-

    Attempt to minimise J(theta_1) where theta_1 is a real number.

See video for graph :)

    theta_1 = theta_1 - alpha * (d/d theta_1) J(theta_1)

We look at the *tangent* at a point on the curve.

We can increment *or* decrement the parameter depending on whether the tangent slope is positive or
negative.

If alpha is too small, gradient descent can be slow.

If alpha is too large, gradient descent can overshoot the minimum. It may fail to converge, or even
diverge.

If we are already at the local minimum, then taking another step will not move us away, as the gradient will be 0.

Gradient descent can converge to a local minimum, even with the learning rate, alpha, fixed.

As we approach a local minimum, gradient descent will automatically take smaller steps. So, no need
to decrease alpha over time. The gradient varies.

## Video: Gradient Descent For Linear Regression ##

Let's compare our gradient descent algorithm with our linear regression model:-

    Gradient Descent Algorithm

    repeat until convergence {
        theta_j = theta_j - alpha * (d/d theta_j) J(theta_0, theta_1)

        (for j=1, j=0)

    ----

    Linear Regression Model

    h_theta(x) = theta_0 + theta_1 * x

    J(theta_0, theta_1) = 1/(2m) * sigma{i=1->m}(h_theta(x^(i)) - y^(i))^2

    Minimise J(theta_0, theta_1).

Let's apply gradient descent to minimise our cost function.

We must determine what the partial derivative term evaluates to.

    d/dtheta_j J(theta_0, theta_1) = d/dtheta_j ( 1/(2m) Sigma{i=1->m}(h_theta(x^(i)) - y^(i))^2 )

    = d/dtheta_j 1/(2m) Sigma{i=1->m}(theta_0 + theta_1 x^(i) - y^(i))^2

We need to determine this term for j=0, and j=1.

The derivation is not shown, but the result is:-

    j=0, d/d(theta_0) J(theta_0, theta_1) = (1/m) * Sigma{i=1->m} (h_theta(x^(i)) - y^(i))
    j=1, d/d(theta_1) J(theta_0, theta_1) = (1/m) * Sigma{i=1->m} ( (h_theta(x^(i)) - y^(i)) * x^(i) )

(This uses multivariate calculus.)

Plugging these values back in to the gradient descent algorithm:-

    repeat until convergence {
        theta_0 = theta_0 - alpha * (1/m) * Sigma{i=1->m} (h_theta(x^(i)) - y^(i))
        theta_1 = theta_1 - alpha * (1/m) * Sigma{i=1->m} ( (h_theta(x^(i)) - y^(i)) * x^(i))
    }

    These should be updated simultaneously.

We don't need to be concerned about local minima here, as the linear regression cost function is
always a 'bowl' shaped curve (see vid @ 05:00), technical term = "convex function". In this
function, local optimum = global optimum.

Let's try this is reality.

Let's initialise the calculation at:-

    theta_0 = 900, theta_1 = -0.1

As we continue we eventually find the optimum. Woo!

This approach is sometimes called "batch" gradient descent. Batch - each step of gradient descent
uses *all* the training examples, i.e. in the summations.

There exists a method for numerically solving for the minimum of the cost function J, without
needing to use an iterative algorithm like gradient descent, however we'll talk about that
later. (This method is called the normal equations method.)

It turns out that gradient descent scales better to larger datasets than the normal equations
method.

## Video: What's Next ##

Two extensions:-

1. Extension 1:-

    In min J(theta_0, theta_1) can solve for theta_0, theta_1 exactly without needing iterative
    algorithm (gradient descent).
    
2. Learn with larger number of features.

    +--------------------+--------------------+--------------------+--------------------+--------------------+
    |   Size (feet^2)    |  No. of bedrooms   |   No. of floors    | Age of home (yrs)  |   Price ($1000)    |
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


Hard to visualise graphically, notation gets more complicated.

Turns out linear algebra is very useful for working with this amount of data.

Linear algebra very useful for implementing efficient solutions to problems.

3. Linear Algebra Review
------------------------

## Video: Matrices and Vectors ##

What are matrices, what are vectors?

Matrix: rectangular array of numbers, e.g.:-

    [ 1402 191  ]    [ 1 2 3 ]
    [ 1371 821  ] or [ 4 5 6 ]
    [ 949  1437 ]
    [ 147  1448 ]

Dimension of matrix = rows x cols, e.g. the above are 4x2 and 2x3.

Sometimes written as |R^(4x2) and |R^(2x3) to refer to all matrices of the specified dimensions.

We refer to elements of a matrix, A, as A_ij, where i = element row, j = element column.

So in:-

    [ 1402 191  ]
    [ 1371 821  ]
    [ 949  1437 ]
    [ 147  1448 ]

    A_11 = 1402
    A_12 = 191
    A_32 = 1437
    A_41 = 147
    A_43 = undefined (error)

Vector is a special case of a matrix - an nx1 matrix.

E.g.:-

        [ 460 ]
        [ 232 ]
    y = [ 315 ]
        [ 178 ] <- 4-dimensional vector.


    |R^4 = set of all 4-dimensional vectors.

    y_i = ith element

    So here y_1 = 460, y_2 = 232, y_3 = 315, y_4 = 178.

There are two different means of indexing into a vector, 1-indexed and 0-indexed. 1-indexed vectors
tend to be more common in mathematics, and this is how we will refer to vector elements most of the
time on the course, however sometimes useful in machine learning to use 0-indexed.

Usually use upper-case to refer to matrices, e.g. A, B, C, X, lower-case to refer to vectors,
e.g. a, b, x, y.

## Video: Addition and Scalar Multiplication ##

How do we add matrices? Add up the elements, e.g.:-

    [ 1 0 ]   [ 4 0.5 ]   [ 5 0.5 ]
    [ 2 5 ] + [ 2 5   ] = [ 4 10  ]
    [ 3 1 ]   [ 0 1   ]   [ 3 2   ]

Can only add matrices of the same dimensions (here 3x2).

To multiply by a scalar, multiply each element by the scalar.

    3*[ 1 0 ]   [ 3 0  ]
      [ 2 5 ] = [ 6 15 ]
      [ 3 1 ]   [ 9 3  ]

Scalar multiplication is commutative.

We can also divide:-

    [ 4 0 ]              [ 4 0 ]    [ 1   0   ]
    [ 6 3 ] / 4 =  1/4 * [ 6 3 ] =  [ 3/2 3/4 ]

Can combine the operations:-

      [1]   [0]   [3]
      [4]   [0]   [0]/3
    3*[2] + [5] - [2]

      [ 3  ]   [0]   [1  ]
      [ 12 ]   [0]   [0  ]
    = [ 6  ] + [5] - [2/3]

      [ 2      ]
      [ 12     ]
    = [ 10 1/3 ]

## Video: Matrix Vector Multiplication ##

E.g.:-

    [ 1 3 ]         [ 1*1 + 3*5 ] = [ 16 ]
    [ 4 0 ] [ 1 ]   [ 4*1 + 0*5 ] = [ 4  ]
    [ 2 1 ] [ 5 ] = [ 2*1 + 1*5 ] = [ 7  ]

Dimensions: 3x2 x 2x1 = 3x1

       A   * x  =  y

    [      ][ ]   [ ]
    [      ][ ] = [ ]
    [      ][ ]

     m x n  nx1 = m-dimensional vector

To get y_i, multiply A's ith rows with elements of vector x, and add them up.

So,

    y_i = Sigma[j=1->m](A_ij * x_j)

E.g.:-

                   [ 1 ]
    [ 1  2  1  5 ] [ 3 ]   [ 1*1  + 2*3  + 1*2 + 5*1 ]   [ 1  + 6  + 2 + 5 ]   [ 14 ]
    [ 0  3  0  4 ] [ 2 ] = [ 0*1  + 3*3  + 0*2 + 4*1 ] = [ 0  + 9  + 0 + 4 ] = [ 13 ]
    [ -1 -2 0  0 ] [ 1 ]   [ -1*1 + -2*3 + 0*2 + 0*1 ]   [ -1 + -6 + 0 + 0 ]   [ -7 ]

E.g. :-

    House sizes:

    2104
    1416
    1534
    852

    h_theta(x) = -40 + 0.25x

Can construct as matrix/vector multiplication:-

      4x2          2x1    =  4x1

    [ 1 2104 ]   [ -40  ]   [ 1*-40 + 2104*0.25 ]   [ h_theta(2104) ]   [ 486   ]
    [ 1 1416 ] x [ 0.25 ] = [ 1*-40 + 1416*0.25 ] = [ h_theta(1416) ] = [ 314   ]
    [ 1 1534 ]              [ 1*-40 + 1434*0.25 ]   [ h_theta(1434) ]   [ 318.5 ]
    [ 1 852  ]              [ 1*-40 + 852*0.25  ]   [ h_theta(852)  ]   [ 173   ]

    e.g. prediction = data matrix * parameters

## Video: Matrix Matrix Multiplication ##

E.g.:-
             [ 1 3 ]
    [ 1 3 2 ][ 0 1 ]
    [ 4 0 1 ][ 5 2 ]

Treat like matrix/vector multiplication:-

             [ 1 ]
    [ 1 3 2 ][ 0 ] = [ 11 ]
    [ 4 0 1 ][ 5 ]   [ 9  ]

            [ 3 ]
    [ 1 3 2][ 1 ] = [ 10 ]
    [ 4 0 1][ 2 ]   [ 14 ]

Then put these together:-

             [ 1 3 ]
    [ 1 3 2 ][ 0 1 ] = [ 11 10 ]
    [ 4 0 1 ][ 5 2 ]   [ 9  14 ]

       2x3     3x2   =    2x2

    A (m x n) x B (n x o) = C (m x o)

The ith column of the matrix C is obtained by multiplying A with the ith column of B (for i = 1, 2,
..., o).

E.g.:-

    [ 1 3 ] [ 0 1 ]   [ 9  7  ]
    [ 2 5 ] [ 3 2 ] = [ 15 12 ]

Another 'neat trick', e.g.:-

   House sizes:

    2104
    1416
    1534
    852

Only this time we have 3 competing hypotheses:-

    1. h_theta(x) = -40  + 0.25x
    2. h_theta(x) = 200  + 0.1x
    3. h_theta(x) = -150 + 0.4x

Can express this as a matrix multiplication problem:-

    Matrix            Matrix

    [ 1 2104 ]   [ -40  200 -150 ]   [ 486   200.1 691.6 ]
    [ 1 1416 ] x [ 0.25 0.1 0.4  ] = [ 314   341.6 416.4 ]
    [ 1 1534 ]                       [ 343.5 353.4 463.6 ]
    [ 1 852  ]                       [ 173   285.2 190.8 ]

Each column in resultant matrix is predicted house prices for respective hypothesis.

Good linear algebra libraries out there to use

## Video: Matrix Multiplication Properties ##

W.R.T scalars, matrix multiplication is commutative (i.e. order of operands can change).

Let A and B be matrices. Then in general, A x B != B x A (not commutative.)

E.g.:-

    [ 1 1 ] [ 0 0 ] = [ 2 0 ]
    [ 0 0 ] [ 2 0 ]   [ 0 0 ]

    [ 0 0 ] [ 1 1 ] = [ 0 0 ]
    [ 2 0 ] [ 0 0 ] = [ 2 2 ]

Also, consider multiplying an mxn matrix, A, by and nxm matrix B:-

    mxn x nxm -> mxm
    nxm x mxn -> nxn

So in this case, i.e. one where it is possible to swap the order of matrices, the two matrices even
have different dimensions.

E.g. in the case of scalars:-

    3 x 5 x 2 can be represented as:-

    3  x 10 or
    15 x 2

So:-

    3x(5x2) = (3x5)x2

Scalar multiplication is "associative", i.e. a(bc) = (ab)c.

This is true of matrices. So we can solve A x B x C thusly:-

Let D = B x C. Compute A x D.
Let E = A x B. Compute E x C.

### Identity Matrix ###

In scalar numbers, 1 is identity. So 1 x z = z x 1 = z, for any z.

Denoted I, or I_(nxn).

Examples of identity matrices:-

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

Informally:-

(zeroes are big)

[ 1
   1   0
    1
     .
      .
   0   .
        1 ]

A   . I   = I   . A   = A

mxn x nxn = mxm x mxn = mxn

So the dimensions of I have to vary here.

Note, AB != BA in general, but if B = I, then AI = IA.

## Video: Inverse and Transpose ##

As mentioned previously, in real numbers, 1 = "identity".

Each real number has an inverse, such that AB = I. So e.g. 3, inverse = 3^-1 = 1/3.

Not all numbers have an inverse, i.e. 0 - 0^-1 is undefined.

### Matrix Inverse ###

If A is an m x m matrix, and if it has an inverse, then:-

    A(A^-1) = (A^-1)A = I.

Note mxm matrix - 'square matrix', as rows = cols. Only square matrices have inverses.

E.g.

           [ 3 4  ]
    A =    [ 2 16 ] 2x2 so inversable.

    A^1 = [ 0.4   -0.1  ]
          [ -0.05 0.075 ]

    [ 3 4  ] [ 0.4   -0.1  ]   [ 1 0 ]
    [ 2 16 ] [ -0.05 0.075 ] = [ 0 1 ] = I_(2x2)

       A           A^-1

Can compute easily in octave:-

    octave-3.2.3:1> A = [ 3 4; 2 16]
    A =

        3    4
        2   16

    octave-3.2.3:2> pinv(A)
    ans =

       0.400000  -0.100000
      -0.050000   0.075000

    octave-3.2.3:3> inverseOfA = pinv(A)
    inverseOfA =

       0.400000  -0.100000
      -0.050000   0.075000

    octave-3.2.3:4> A*inverseOfA
    ans =

       1.00000   0.00000
      -0.00000   1.00000

    octave-3.2.3:5> inverseOfA*A
    ans =

       1.0000e+00  -4.4409e-16
       2.7756e-17   1.0000e+00

Some matrices don't have inverses, e.g.:-

        [ 0 0 ]
    A = [ 0 0 ]

Intuition: don't have an inverse if 'too close to zero'

Matrices that don't have an inverse are "singular" or "degenerate"

### Matrix Transpose ###

E.g.:-

          [ 1 2 0 ]
    A =   [ 3 5 9 ] 2x3

    A^T = [ 1 3 ] = B
          [ 2 5 ]
          [ 0 9 ]  3x2

Flipping along a 45 degree axis. Or swapping rows + columns!

Let A be an mxn matrix, and let B = A^T.

Then B is an nxm matrix, and Bij = Aji.
