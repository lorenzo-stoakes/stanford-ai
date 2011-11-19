AI/ML Meetup - AI Notes
=======================

12/10/2011
----------

Wetware
Formula
Intelligent Agent

             *------------- sensors       
             v
    *----------------*                          ^
    | agent          |                          |
    | -----          |                          |
    |                |                <--- environment --->
    | control policy |                          |
    *----------------*                          |
            |                                   v
            |
            ---------------> actuators

* Observability - fully-observable vs partially-observable.

Fully-observable - can see everything.
Partially-observable - can't see everything.

* Deterministic vs. stochastic - whether random element involved.

* Discrete vs. continuous - Whether infinite outcomes or not?

* Benign vs. adversarial - does environment interact competitively with you?

Must be deterministic and discrete.

P:  { INITIAL STATE ~ I
|   { actions:        S -> { a }
|   { result:         (s, a) -> S'
|   { Goal test:      S -> bool
|   { Path cost:      (S, [a']) -> N
|   { Step cost:      (s, a) -> N


            (I)
            /\
           /  \
          /\   \
         /  \   \
       (O)   o   o
           etc.

Different subsets of nodes:-

* Seen already
* Not visited yet
* Frontier - Ones about to visit

Begin with:-

    Frontier = { initial }
    Loop:
        If frontier is empty FAIL
        Path = choose(frontier)
        s = If GOAL RETURN [S]

16/11/2011
----------

We have Bayes:-

    [; P(a|b) = \frac{P(b|a)P(a)}{P(b)} ;]
    [; P(a|b) = \frac{P(a \wedge b)}{P(b)} ;]
    
