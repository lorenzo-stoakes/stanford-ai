AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 14 - Game Theory
---------------------

## Introduction ##

Looking at a book called 'Introduction to the Theory of Games' by McKinsey, published in 1952, 4
years before the start of AI. Game theory and AI have grown up together, taking different paths, but
now are beginning to merge.

We've been talking so far about games where players take turns to make moves, which game theory
handles perfectly well, however it can also handle games where the moves are performed
simultaneously, e.g. both players move but player 2 is not aware of player 1's move. The problem
becomes partially observable. The back and forth of trying to determine what move to make given what
you think the opponent might do, and taking into account what they might think of your move.

We're going to consider how this all relates to AI. This is divided into two aspects - agent design,
i.e. given a game, what is the optimal policy, and secondly - mechanism design - given utility
functions, how can we design a mechanism such that when agents act rationally the global utility
will be maximised.

## Dominant Strategy Question ##

Game theory is defined as the study of finding the optimal policy when that policy can depend on the
opponent's policy, and vice-versa.

Let's examine the *prisoner's dilemma* - consider Alice and Bob, both caught at a scene of a crime,
but the police don't quite have the evidence to put them away. The police offer them a deal saying -
if you testify against your cohort then you'll get a reduced sentence.

This is a non-zero-sum game.

Let's draw a matrix of possible outcomes:-

               | A: testify   | A: refuse  |
    -----------+--------------+------------+               
    B: testify |  A=-5, B=-5  | A=-10, B=0 |
    -----------+--------------+------------+
    B: refuse  |  A=0, B=-10  | A=-1, B=-1 |

What strategy will Alice and Bob choose?

Firstly we need to consider a *dominant strategy* - this is a strategy which is better than any
other strategy, no matter what the other player does.

Both Alice and Bob's dominant strategy is to testify, since no matter what the other player does,
testify outdoes refuse - e.g. -5 vs -10 and 0 vs -1.

## Pareto Optimal Question ##

The Pareto optimal outcome is one where there's no other outcome that all players would prefer,
i.e. in the case of the prisoner's dilemma - both refusing where A=-1, B=-1 - this is the ideal case which both players can agree on.

## Equilibrium Question ##

An equilibrium is one in which no player can benefit from switching to another strategy, assuming
the other player stays the same. Famous result from John Nash which proves every game has at least
one equilibrium point.

In the prisoner's dilemma the testify/testify case is an equilibrium point, as if A switches
strategy, they do worse, and if B switches strategy they do worse. We consider each in turn,
assuming the other stays the same.

Clearly here the Pareto optimal is different from the equilibrium state. Both players acting
rationally will move to the equilibrium, despite it being suboptimal.

## Game Console Question 1 + 2 ##

Consider a game console manufacturer acme, and a game manufacturer best. Both have the choice of
targeting blu-ray or dvd. The matrix is:-

           |    A: blu    |   A:dvd    |
    -------+--------------+------------+               
    B: blu |  A=+9, B=+9  | A=-4, B=-1 |
    -------+--------------+------------+
    B: dvd |  A=-3, B=-1  | A=+5, B=+5 |

Is there a dominant strategy for either A or B, and is there an equilibrium?

The two dominant strategies are A=blu, B=blu and A=dvd, B=dvd. There is no dominant strategy for
either player.

The Pareto optimal outcome is A=+9, B=+9.

## 2 Finger Morra ##

It's easy to determine the solution to a game if there is a dominant strategy or a Pareto optimal
strategy.

Let's consider 2 finger morra where each player raises one or two fingers, with one looking for an
odd sum of fingers raised and the other looking for an even sum of fingers raised. The matrix is:-

Note this is a zero-sum game so we don't specify both values each time, as they are equal and opposite.

           | O: one | O: two |
    -------+--------+--------+               
    E: one |  E=+2  |  E=-3  |
    -------+--------+--------+
    E: two |  E=-3  |  E=+4  |

There is no dominant strategy or Patero optimal scenario for this game. There is no single move
which is the best for each player, however there is a 'mixed strategy' which can be used.

A single strategy which involves playing a move or another is known as a 'pure strategy', a mixed
strategy is where you have a probability distribution over the possible moves.

## Tree Question ##

We can examine the 2 finger morra game as a game tree with even as the max player and odd as the min
player:-

<img src="http://codegrunt.co.uk/images/ai/14-tree-question-1.png" />

This is on the assumption of the even player going first, the question here is concerned with the
odd player going forward:-

<img src="http://codegrunt.co.uk/images/ai/14-tree-question-2.png" />

## Mixed Strategy ##

The problem with the minimax solution given previously is that both odd and even are handicapped
severely by having to give away their entire strategy before the other player takes their go.

Let's consider a scenario where the players aren't quite as open, i.e. defining a given move for a p
probability from which we can calculate expectation, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/14-mixed-strategy-1.png" />

## Solving the Game ##

What value of p should E choose? We know that O will pick the smallest of the two terminal states,
so the optimum is likely to come from the situation where both one and two choices by O are equal to
one another, e.g.:-

    [; 2p - 3(1-p) = -3p + 4(1-p) ;]

Thus:-

    [; p = \frac{7}{12} ;]

We can do a similar analysis for q, where we determine that:-

    [; q = \frac{7}{12} ;]

Also.

In each case the utility to E is equal to [; \frac{-1}{12} ;], so we have solved the game.

## Mixed Strategy Issues ##

The introduction of mixed strategy raises some interesting philosophical problems related to:-

* Randomness
* Secrecy
* Rationality

If we find an optimal mixed strategy, then counterintuitively it's perfectly fine to reveal to the
opponent our *overall* strategy, i.e. the 'grand strategy' of how we do things. However, it is not
ok to tell our opponent what we are going to do *specifically* in a given application of the
strategy.

With respect to rationality, we say a rational agent is one which does the 'right thing', which is
still true, however there are games in which you can do better if your opponent thinks you are *not*
rational. This has been said about various politicians throughout history. E.g. - if one country
threatens another with war, and it is irrational for either side to open hostilities, then in a
usual case the other country would not see the threat as credible. However if the threatening leader
has cultivated a reputation as being irrational, then the threat can be considered more credible.

Being irrational doesn't help, seeming irrational might.

## 2x2 Game Question 1 + 2 ##

Consider the following:-

           | N: one | N: two |
    -------+--------+--------+               
    M: one |  M=+5  |  M=+3  |
    -------+--------+--------+
    M: two |  M=+4  |  M=+2  |

Where M is max and N is min.

Here, if M goes first:-

    [; 5p + 4(1 - p) = 3p + 2(1 - p) ;]
    [; 5p + 4 -4p = 3p + 2 - 2p ;]
    [; p + 4 = p + 2 ;]

So we can't choose a p such that min's decision makes no impact

The same goes for the case where N goes first:-

    [; 5p + 3(1-p) = 4p + 2(1-p) ;]
    [; 5p + 3 - 3p = 4p + 2 - 2p ;]
    [; 2p + 3 = 2p + 2 ;]

In fact here we don't need to use a mixed strategy, as both players have a dominant strategy - max's
is one as, no matter what min does, it is better for max, and 2 is better for min as no matter what
min does strategy 2 does better (i.e. minimises max's utility).

So the probability is p=1 for max choosing 1, p=0 for max choosing 2, p=1 for min choosing 2, p=0
for min choosing 1. This means we always end up in the M=+3 case, hence the utility for max is
always 3.

Consider the following:-

           | N: one | N: two |
    -------+--------+--------+               
    M: one |  M=+3  |  M=+6  |
    -------+--------+--------+
    M: two |  M=+5  |  M=+4  |

No dominant strategy here.

Let's consider a mixed strategy, max going first:-

    [; 3p + 5(1-p) = 6p + 4(1-p) ;]
    [; 3p + 5 - 5p = 6p + 4 - 4p; ]
    [; -2p + 5 = 2p + 4 ;]
    [; 4p = 1 ;]
    [; p = \frac{1}{4} ;]

Min going first:-

    [; 3p + 6(1 - p) = 5p + 4(1 - p) ;]
    [; 3p + 6 - 6p = 5p + 4 - 4p ;]
    [; -3p + 6 = p + 4 ;]
    [; 4p = 2 ;]
    [; p = \frac{1}{2} ;]

And the utility is [; -2 * \frac{1}{4} + 5 = 4.5 ;]

## Geometric Interpretation ##

Looking back to the 2 finger morra game:-

<img src="http://codegrunt.co.uk/images/ai/14-geometric-interpretation-1.png" />

Each side is trying to maximise/minimise E's utility, and both end up at the same value.

## Poker ##

So far we've dealt with games which take a single turn - both players take a turn and simultaneously
reveal their moves, and then the game is complete.

Game theory can also deal with more complex games where there can be multiple rounds.

We consider a simplified version of poker where we have a deck of only KKAA, and two rounds - in the
first the player can raise/check, and in the second they can call/fold:-

<img src="http://codegrunt.co.uk/images/ai/14-poker-1.png" />

Similar to the game tree covered in the previous unit.

This is known as the 'sequential game format', in 'extensive form'.

Keeps a track of what each agent knows or doesn't know - but the agents aren't aware at which point
in the tree they are.

We can represent it in the 'normal form' i.e. the matrix form we've seen previously, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/14-poker-2.png" />

Here we have two equilibria - 1:rk, kk; 2:cf

In real poker the table would have [; 10^{18} ;] states and would be impossible to deal with. We
need to have a way to bring the states down to something reasonable.

## Game Theory Strategies ##

A great strategy is abstraction - instead of dealing with every possible state of the game, we take
similar states and treat them as if they were the same. For example in poker a strategy which works
well is to eliminate suits - if the player is not trying to get a flush then it doesn't matter.

Can lump similar cards together, e.g. if I'm holding a pair of tens, can think of opponent's cards
as those which are lower than ten, equal to ten, or greater than ten. Can lump bets into say small, medium and large.

Another abstraction is, rather than considering every deal of the cards, consider a small subset of
the deals using a monte carlo approach of the subset.

Game theory works quite well with:-

* Uncertainty
* Partial observability
* Multiple agents
* stochasticity 
* Sequential moves
* Dynamic (?)

But not so good with:-

* Unknown actions - Need to know what all the actions are before we can apply it.
* Continuous actions - Can't reduce continuous actions to a matrix.
* Irrational opponents - Doesn't tell us how to exploit irrational opponents, only how to deal with
  rational ones.
* Unknown utilities - If we don't know what to optimise game theory can't help us.

## Fed vs. Politicians Question ##

Nothing to add.

## Mechanism Design ##

'Mechanism design' could also be known as game design - somebody is designing a game for others to
use, need to set up the rules such that there is a high expected outcome for the people that run the
game, players who play the game + the public at large.

An example of a game is the advertising game, e.g. google ads.

The aim of the game is to make the auction of ads to be appealing to bidders for the ads and those
who want to use them.

One property you'd like an auction to have is to attract as many bidders as possible. One aspect of
this is to make less work for them. It's easier for the bidders if they have a dominant strategy -
far easier to solve a game if you have one vs. a mixed strategy. This means you don't even need to
think about what the other participants are doing - also known as truth revealing or incentive-compatible. 

## Auction Question ##

Let's discuss a type of auction known as a 'second price' auction. This is popular in various
internet search and auction sites.

The way it works is that people bid, and the winner is the highest bidder, however the price is set
as the second highest price.

Let's say the value you're bidding on is v, your bid is b and the highest other bid is c. The payoff
if you win is v-c, otherwise it's 0 i.e. you lose the auction and there is no cost:-

    if b>c: v-c
      else: 0

Note that if there are ties in a policy this defines whether a dominant strategy is a 'strictly'
dominant policy, or a 'weakly' dominant one.

Let's examine the following grid:-

            | c=7  | c=9  | c=11 | c=13 |
            +------+------+------+------+
    b = 12  |  3   |  1   |  -1  |   0  |
            +------+------+------+------+    
    b = 10  |  3   |  1   |   0  |   0  |
            +------+------+------+------+    
    b = 8   |  3   |  0   |   0  |   0  |
            +------+------+------+------+

Clearly the dominant strategy is b=10. This is weakly dominant.

Looking at this - if you bid above the true value, but an opponent bid just underneath, you'll pay
over the odds. However if you bid too low, the opponent could sneak in above you and steal the
auction.

'Truth-revealing' because it encourages the true value to be chosen.
