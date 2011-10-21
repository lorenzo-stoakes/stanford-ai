AI/ML Meetup
============

28/9/2011
---------

Probability
-----------

Real Basics
-----------

* Trials - repeatable experiment, e.g. throwing a dice.

Omega = set of all outcomes, e.g. 1,2,3,4,5,6 results of roll of dice.

* Event = a subset of these, e.g. rolling an even number, rolling <= 2, etc.

* Event space, E = power set of Omega, i.e. all possible outcomes.

* E |-> [0, 1] <- range of possible results.

* Probability of whole event space = 1.

* Clearly a uniform distribution of probability 1/6 each.

* P(A) = |A| / |Omega|

* |A| = Size of set of events (space) we're interested in, e.g. single dice roll = 1.

* |Omega| = Size of set of all events (event space?) = 6.

* P(1U2) = P(1) + P(2) where these events are independent.

* P(1n1) = P(1) * P(1).

* Conditional probability: P(A|B) = P(AnB) / P(B) where P > 0.

* A = >= 2 heads, B = throw head first. P(B) = 0.5. P(A|B) = P(AnB) / P(B) = 3/8 / 1/2 = 6/8 = 3/4

* If P(A|B) = P(A) then A and B are conditionally independent.

* E.g. A = throwing a head second, B = throwing a head first, so P(A|B) = 2/8 / 1/4 = 1/2.

    HHH
    HHT
    HTH
    HTT
    THH
    THT
    TTH
    TTT

* Union + intersection commutative, but clearly not conditionality!

* Bayes: P(B|A) = P(A|B) * P(B) / P(A).

* Prior probability = B, typically, because we already know about this.

Tim's:- UUUUS
Hry's:- UUS

* E.g.: Yaks. Shaven or not, Tim/Henry's. A = Yak is shaved, B = Belongs to Henry from entire
  universe of Yaks.

* P(A) = 2/8 = 1/4

* P(B) = 3/8

* P(A|B) = 1/3

* P(B|A) = 1/3 * 3/8 / 1/4 = 4/3 * 3/8 = 1/2

           Tim         |   Henry

            5          |     3
    <----------------->|<---------->
                     1 | 1
    +---------------+---+---+------+
    |               |###|###|      |
    |               |###|###|      |
    |               |###|###|      |
    |               |###|###|      |
    |               |###|###|      |
    |               |###|###|      |
    |               |###|###|      |
    |               |###|###|      |
    |               |###|###|      |
    +---------------+---+---+------+

* E.g. breast cancer.

A = M = Positive mammography.
B = C = Probability of having cancer in the age group. P(B) = 0.01.

Probability of positive mammography if has cancer = P(A|B)    = 0.8.
Probability of positive mammography if not cancer = P(A|notB) = 0.096.

P(B|A) = probability of cancer if positive mammography.

P(B|A) = P(A|B) * P(B) / P(A) = 0.96 * 0.01 / P(A).

P(A)?!!

* A = (A n B) u (A n notB)
* P(A) = P(A n B) + P(A n notB)
* P(A) = P(A|B) * P(B) + P(A|notB) * P(notB)

P(A) = 0.8 * 0.01 + 0.096 * 0.99 = 0.008 + 0.09504 = 0.10304

Hence P(B|A) = 0.8 * 0.01 / 0.10304 = 0.008 / 0.10304 = 0.0776.

* Random variable, X, can take any value, quantifies the event in question, e.g. people's heights.

* Can be any number. Know an observed value (?), random variable is what we're looking
  at. Probability of random variable falling in a range.

* f: event -> number.

* PDF = Probability Density Function for continuous values.

E.g.:-

        ^
        |    x
        |   x x
      P |  x   x
        | x     x
        |x       x
        *------------->
           Rainfall

Intuition: 'area under curve'.

* Integral of entire PDF = 1, certainty.

* EV = mean?

5/10/2011
---------

Linear Algebra
--------------

Determinants
------------

determinant = ad - bc for 2x2 -

    [ a b ]
    [ c d ]

Also inverse =
    1/(ad-bc) * [  d -b ]
                [ -c  a ]

If ad-bc = 0 then 'singular', can't find inverse.

Using Matrices to Solve Linear Systems
--------------------------------------

 x - 2y = 1
3x + 2y = 11

Can use graphs to solve :-)


   +
  -+
   |\-                     /-
   |  \-                 /-
   |    \-            /--
   |      \-        /-
   |        \-   /--
   |----------X--
   |        /-| \-
   |     /--  |   \-
   +---/------------\----------+

We want the point marked x here.

This can be represented as a matrix:-

    [ 1 -2 ] [x]   [1]
    [ 3  2 ] [y] = [11]

If we have

 x +  y +  z = 6
 x + 2y + 3z = 4
 x + 5y - 3z = 2


We can look at these like:-

    [ 1  1  1 ] [x] = [6]
    [ 1  2  3 ] [y] = [4]
    [ 1  5 -3 ] [z] = [2]

We subtract a row (or multiples of that) from another to obtain sufficient zero coefficients.

We want to get an upper triangular form...

    [ 1  1  1 | 6 ]
    [ 1  2  3 | 4 ]
    [ 1  5 -3 | 2 ]

    Subtract first row:-

    [ 1  1  1 | 6  ] 
    [ 0  1  2 | -2 ]
    [ 0  4 -4 | -4 ]

    Subtract 2nd row * 4:-

    [ 1 -3 -7   | 14 ]
    [ 0  1  2   | -2 ]
    [ 0  0  -12 |  4 ]

So:-

    x - 3y - 7z = 14
    y + 2z = -2
    -12z = 4

    z = -1/3
    y = -2 + 2/3 = -1 1/3
    x = 14  - 4 - 7/3
      = 7 2/3

    x = 7 2/3
    y = -1 1/3
    z = -1/3

Eigenvector - doesn't change direction after matrix transformation.

So Ax = lambda . x, where lambda is a scalar value.

Eigen value = degree by which it scales...?

A . x = lambda . x

If A =

    [ 1 2 ]
    [ 4 3 ]


    A . x = lambda . x
    A . x = lambda . I . x

    (A - lambda . I) . x = 0
    (lambda . I - A) . x = 0

    det(lambda .  I - A) = 0

    [ L 0 ]   [ 1 2 ]
    [ 0 L ] - [ 4 3 ] =

    [ L-1 -2  ]
    [ -4  L-3 ]

    det of above = (L-1)*(L-3) - 8 = 0 So (L-1)(L-3) = 8 => L^2 - 4L + 3 = 8 => L^2 - 4L - 5 = 0 => (L+1)(L-5) So L = -1, 5

    1x + 2y = -x
    4x + 3y = -y

    => 2x + 2y = 0
       4x + 4y = 0

    => x + y = 0, y = -x


