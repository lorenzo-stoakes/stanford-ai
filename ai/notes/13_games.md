AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 12 - Games
---------------

## Introduction ##

In this unit we're going to be talking about games. Why games - a. they're fun, b. form a
well-defined subset of the world, as opposed to e.g. an autonomous plane which needs to worry about
everything in the world. Form a small scale model of a specific problem - namely dealing with adversaries.

## Technologies Question ##

What do the following technologies most address?

* MDPs - Stochastic environment
* POMDPs, Belief Space - Partially observable environment
* Reinforcement Learning - Unknown environment
* A*; h function; Monte Carlo - Computational limitations

## Games Question ##

Wittgenstein asserted that games aren't quite as easily definable as you might think, rather there
are a set of overlapping criteria as to the definition of a game. Let's consider the following games
and which of the components of stochastic (St), partially observable (PO), unknowns in the
environment (U), or adversarial (A) - though subjectively - apply:-

* Chess, Go - A
* Robot Football - St, PO, A
* Poker - PO, A
* Hide + Seek - St, PO, U, A
* Cards Solitaire - PO
* Minesweeper - PO

PN's argument for stochasticity being a factor in only robot football and hide + seek is based on
the fact that movements aren't guaranteed to result in what is intended, e.g. if you wish to move
forward one meter, you might in fact not.

You might easily consider games like poker/solitaire to be stochastic (as did LS) due to not knowing
what a downturned card will be when flipped over, however PN has classed this as partial
observability (PO) since you aren't picking a card randomly, rather the cards are arranged in an
order which you are simply unaware of.

Only hide + seek has unknowns in the environment, arguably because somebody might hide in a room you
didn't know about beforehand.

Less controversially, only cards solitaire and minesweeper are considered not to be
adversarial. You could consider something like cards solitaire to be adversarial since you are
effectively 'playing against the game', however the game isn't actively trying to defeat you - in
this context 'adversarial' is taken to mean that the opponent is taking into account your thinking
when performing their thinking in order to defeat you.

## Single Player Game ##

Looking again at the sliding block puzzle game we can see that it is:-

* Single player
* Deterministic

We can solve this using search through a state space:-

<img src="http://codegrunt.co.uk/images/ai/13-single-player-game-1.png" />

How do we describe the game?

* Set of states, [; S ;], with initial state [; S_0 ;].
* Player, [; P ;].
* Set of actions, [; Actions(S, P) ;] - sometimes we can determine this purely from the state,
  otherwise we need to know about the player too.
* Transition function - [; Result(s, a) \rightarrow s' ;]
* Terminal test (is it the end of the game?) - [; Terminal(S) \rightarrow T, F ;]
* Terminal utilities - for a given state + player what is the value of the game to that player? -
  [; U(S, P) \rightarrow $ number$ ;] - in simpler games it's win or loss (possibly 1 or -1).

## Two Player Game ##

Let's consider games like chess and checkers which are:-

* Deterministic
* Two-player
* Zero-sum - the sum of the utilities of the game to the two players is equal to 0, e.g. +1 to one
  player, -1 to the other.
  
We consider two players who we name max and min because max wants to maximise their utility, and min
wants to minimise max's utility (minimax). Consider a search tree:-

<img src="http://codegrunt.co.uk/images/ai/13-two-player-game-1.png" />

Here upwards-facing triangles are max, downwards-facing are min and terminal states are squares.

Note that the +1 and -1 scores are max's.

We can determine the values of all the states from the terminal ones, on the assumption that players
are rational and the game is finite (i.e. the game terminates after a finite number of steps), by
working backwards and considering whose move it is - if max's then the move which results in +1 will
be taken, if min's then the move resulting in -1 will be taken (on the assumption that results
propagate upwards from terminal states).

We've taken a utility function which was only defined on terminal states and made it applicable to
all other states.

## Two Player Function ##

Let's define a function which tells us how to compute the value for a given state:-

<img src="http://codegrunt.co.uk/images/ai/13-two-player-function-1.png" />

In theory should answer any two-player deterministic finite game.

## Time Complexity Question ##

We need to determine the complexity of the algorithm just given to determine whether it's feasible
to use it.

Let's declare:-

* b - the average branching factor, i.e. the average number of actions (or moves) coming out of a
position.
* m - depth of the search tree.

Then the time complexity of the search is [; O(b^m) ;]

## Space Complexity Question ##

Space complexity is [; O(bm) ;] as to examine a given path we only need to consider the branches of each
node for each level.

## Chess Question ##

Assuming b = 30 and m = 40, that your program can evaluate 1bn nodes per second, and that you are
leant every computer in the world - how long will it take?

Since time complexity is [; O(b^m) ;], and [; 30^40 \simeq 10^59 ;] we can divide by billions all
day long and still be left with a huge number so time required would be several times the lifetime
of the universe.

## Complexity Reduction Question ##

How do we deal with the complexity of [; b^m ;]?

* Reduce b
* Reduce m
* Convert the tree to a graph

## Review Question ##

Determine values in min/max tree using the algorithm given previously. No annotations required.

## Reduce B + Question ##

How can we reduce the branching factor?

By examining the tree from the previous question, we can determine whether it's worth evaluating
branches where it's simply a path which is not going to get evaluated:-

<img src="http://codegrunt.co.uk/images/ai/13-reduce-b-question-1.png" />

Here we can see that, once we've seen the terminal node of 2 under the second child node of the
root, we can determine that since the node above is a minimising one, it will pick the smallest
value and thus it will have value of 2 or less. Since the root node is a maximising node, it will
always prefer the first child's 3 value to the second child's 2 or lower values, so there's simply
no need to evaluate its children further after seeing the two. We *prune* these branches.

In this case, the savings weren't great since the nodes we're removing from evaluation are terminal
nodes, however in real scenarios it can be significant since we could be removing large numbers of
child nodes beneath a pruned node.

Repeating for the right-most child of the root:-

<img src="http://codegrunt.co.uk/images/ai/13-reduce-b-question-2.png" />

## Reduce M ##

Considering a huge tree, how do we reduce its depth?

The first approach is to simply cut the tree at a given depth by fiat, essentially pretending that
at a certain depth all the nodes are terminal nodes:-

<img src="http://codegrunt.co.uk/images/ai/13-reduce-m-1.png" />

Since the game hasn't actually finished at the arbitrary cut-off point, we need to determine some
way to determine values of nodes at this point. This can be achieved with an *evaluation function*
parameterised by state. We want it to be stronger for positions which are stronger, and weaker for
positions which are weaker.

How do we determine this evaluation function? Firstly, we could get it simply from experience - from
looking at previous games with similar positions and working out what their values were. We could
try to break that down into components using something we know about the game, e.g. chess values
(pawn=1, knight/bishop=3, rook=5, queen=9).

This could be achieved with a weighted sum such as:-

    [; ev(s) = \sum w_i p_i ;]

Where [; p_i ;] is the count of pieces of type i. We'd have positive weights for the player's pieces
and negative weights for the opponent's pieces. We could add other 'features' (similar to machine
learning) such as (in chess) control of the centre, the presence or absence of a doubled-pawn,
etc. - we could use machine learning to determine weightings.

## Computing State Values ##

We adjust our value function to reduce b and m:-

<img src="http://codegrunt.co.uk/images/ai/13-computing-state-values-1.png" />

We've added bookkeeping values here:-

* Depth - the current depth which we track simply by incrementing as we traverse the tree,
* [; \alpha ;] - The best value for MAX along the path we're currently exploring,
* [; \beta ;] - The best value for MIN along the path we're currently exploring.

This relates to the name of the algorithm we are using to reduce b - alpha/beta pruning.

Initially we start with:-

    [; value(s_0, 0, -\infty, +\infty) ;]

Our maxValue function is updated as follows:-

<img src="http://codegrunt.co.uk/images/ai/13-computing-state-values-2.png" />

## Complexity Reduction Benefits ##

We've asserted that we can reduce complexity via the following 3 approaches - let's assess the
impact of each:-

* Reduce b - we use the [; \alpha ;] - [; \beta ;] pruning technique as discussed. This reduces time
  complexity from [; O(b^m) ;] to [; O(b^{m/2}) ;]. We get a different level of pruning depending on
  which order we prune branches in - if we pick a 'good' branch first then we prune more than if we
  pick a bad one.
* Reduce m - This, unlike the other two approaches, means we are losing information and can get into
  trouble, since we're essentially giving up on finding a deterministic answer at a certain depth.
* Translate the search tree into a graph - e.g. chess - we can have opening books which specify
  correct moves originating from multiple positions so we can represent this as a graph rather than
  a tree. We can do this with closing (i.e. endgame) moves too, since with a few number of pieces we
  can deterministically work out the optimal move. The so-called 'killer-move' heuristic can be
  useful too - if a move occurs in one branch which has a big impact on the game (e.g. opponent
  taking a queen), then it's worth checking sister branches for the move too.

## Pacman Question ##

Consider the following search tree:-

<img src="http://codegrunt.co.uk/images/ai/13-pacman-question-1.png" />

We cut off the search at the bottom of the tree shown.

## Chance + Question ##

We want to deal with randomness (stochasticity) in games?

Let's update the value function to handle a 'chance node':-

<img src="http://codegrunt.co.uk/images/ai/13-chance-question-1.png" />

Here we determine the expected value rather than min or max at chance nodes, e.g. rolling a dice:-

<img src="http://codegrunt.co.uk/images/ai/13-chance-question-2.png" />

## Terminal State Question ##

Nothing to note.

## Game Tree Question 1 + 2 ##

Nothing to note.

## Conclusion ##

We started by looking at 1 player deterministic games with max states and terminal states, and
determined that it was essentially the same as typical search.

We then added another player for 2-player adversarial games using minimax, then optimised using a
cutoff/evaluation function which is an estimate, before using alpha/beta pruning to reduce branches
we have to examine.

Finally, we added handling for stochasticity which uses expected values to make decisions.
