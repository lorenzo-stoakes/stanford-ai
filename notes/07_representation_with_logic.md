AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 7 - Representation with Logic
----------------------------------

## Introduction ##

We've seen that:-

* AI is about managing complexity and uncertainty.
* Search can discover sequences of actions to solve problems.
* Probability theory can represent uncertainty and help us reason with it.
* Machine learning can be used to learn and improve.

AI is a big field because we're pushing against complexity in at least 3 directions:-

1. Agent design - Moving from reflex-based to goal-based and utility-based agents.

2. Complexity of environment - Start with simple environments, then have to take into account
partial observability, stochastic actions, multiple agents, etc.

3. Representation - the agent's model of the world becomes increasingly complex.

In this unit we're looking at representation - how the tools of logic can be used by an agent to
better model the world.

## Propositional Logic ##

Let's look at an example, the alarm problem:-

Propositional symbols:-

    B, E, A, M, J - Burglary, Earthquake, Alarm going off, Mary calling, John calling

These can be true or false.

Degree of belief is not a number, rather we consider variables to be true, false or unknown.

We can make logical sentences using these symbols and the logical constants true and false.

We combine them together using logical operators.

We can express the idea of the earthquake or burglary occurring causing the alarm as follows:-

    [; (E \vee B) \Rightarrow A ;]

Where:-

    [; \vee ;] = or
    [; \Rightarrow ;] = implies

We can express the idea of both John and Mary calling when the alarm is going off as:-

    [; A \Rightarrow (J \wedge M) ;]

Where:-

    [; \wedge ;] = and

If we want to express the idea that John calls if, and only if Mary calls, we can write that as:-

    [; J \Leftrightarrow M ;]

Where:-

    [; \Leftrightarrow ;] = equivalent

If we wanted to express the idea that John calls when Mary doesn't and vice-versa, we can write that
as:-

    [; J \Leftrightarrow \lnot M ;]

A propositional logic sentence is either true or false with respect to a model of the world. A model
is just a set of true/false values for all the values, e.g.:-

    { B: true, E: false, ... }

We can use truth tables to assess the truth of a sentence with respect to a model.

## Truth Tables ##

E.g.:-

    +----------+----------+----------+----------+----------+----------+----------+
    |    P     |    Q     |    ¬P    |  P ∧ Q   |  P ∨ Q   |  P ⇒ Q   |  P ⇔ Q   |
    +----------+----------+----------+----------+----------+----------+----------+
    |  false   |  false   |   true   |  false   |  false   |   true   |   true   |
    +----------+----------+----------+----------+----------+----------+----------+
    |  false   |   true   |   true   |  false   |   true   |   true   |  false   |
    +----------+----------+----------+----------+----------+----------+----------+
    |   true   |  false   |  false   |  false   |   true   |  false   |  false   |
    +----------+----------+----------+----------+----------+----------+----------+
    |   true   |   true   |  false   |   true   |   true   |   true   |   true   |
    +----------+----------+----------+----------+----------+----------+----------+

Similar to use of concepts in english, only 'or' is quite ambiguous in english, and here it refers
to inclusive or, i.e. we care about whether either P or Q are true, or both.

## Question 1 + 2 ##

Nothing to add.

## Terminology ##

* A *valid* sentence is true in any model, in any possible combination of values of the
propositional symbols.

* A *satisfiable* sentence is true in a given model, but not necessarily all of them.

* An *unsatisfiable* sentence is not true in any model.

## Propositional Logic Limitations ##

Propositional logic is a powerful language for what it does. There are very efficient inference
mechanisms for determining inference and satisfiability (not discussed).

It has a few limitations:-

* It has no capability for handling **uncertainty** - it can only cope with true/false values.
* We can only talk about events that are true/false in the world. We can't talk about objects that
  have properties, e.g. size, weight, colour, etc. nor can we talk about the relations between
  objects.
* There are no shortcuts to succinctly talk about a lot of different things happen, e.g. if we had a
  vacuum world with a thousand locations, and we wanted to say every location was clean, we'd need a
  conjunction of 1,000 propositions. Couldn't just have a single sentence indicating that this is
  the case.

First order logic addresses the latter 2 limitations.

## First Order Logic ##

We'll have a look at first order logic and also how it relates to the other logics we've seen, e.g.
propositional logic and probability theory.

We'll talk about what they say about the world, i.e. the ontological commitment of these logics, and
what types of beliefs agents can have using these logics, which is known as the epistemological
commitments of the logics, e.g.:-

    +------------------------------+------------------------------+------------------------------+
    |            Logic             |            World             |           Beliefs            |
    +------------------------------+------------------------------+------------------------------+
    |      First-Order Logic       |Relations, Objects, Functions |            T/F/?             |
    +------------------------------+------------------------------+------------------------------+
    |     Propositional Logic      |            Facts             |            T/F/?             |
    +------------------------------+------------------------------+------------------------------+
    |      Probability Theory      |            Facts             |      [0..1] real number      |
    +------------------------------+------------------------------+------------------------------+

Logics vary both in what you can say about the world, and what you can believe about things you've
said about the world.

Let's look at some different ways of looking at representations of the world:-

* Atomic - another way to look at representation is to break the world up into *atomic*
representations. This means a representation of the state is simply an individual state, with no
pieces inside of it. We've used this for search and problem solving, e.g. we could look at states
and determine whether they are identical, or perhaps a goal state, but there wasn't any internal
structure to the states.

* Factored - In propositional and first-order logic we *factored* the world up into several
  variables, e.g. via B, E, A, M, J. These don't necessarily have to be boolean.

* Structured - In a *structured* representation, an individual state is not just a set of values for
  variables, it can contain relationships between objects, e.g. programming languages/databases.

## Models ##

How does first-order logic work? What does it do? We start with a model.

Models are more complex than propositional logic.

Suppose we look at 4 scrabble tiles:-

    A(1) C(3)
    B(3) D(2)

We can define constants: A, B, C, D, 1, 2, 3.

We don't have to have a 1-to-1 relationship between constants and objects, we can have more than one
constant which refers to the same object, or perhaps refers to no object at all.

Functions are defined as mappings from object to object. E.g. we might have a function:-

    NUMBEROF { A->1, B->3, C->3, D->2 }

In addition, we can have 'relations', e.g.:-

    ABOVE: { [A, B], [C, D] }

    VOWEL: { [A] }

    RAINY: { } { [] }

Where {} denotes a set and [] tuples.

Here 'rainy' doesn't refer to any of the objects, it's a separate consideration. The arity of rainy
is 0, so when true then we have a single empty tuple, when false we have the empty set.

## Syntax ##

In first order logic we have sentences (like propositional logic) and terms, which describe objects,

Sentences are predicates which correspond to relations.

Note: Here 2=2 denotes the equality relation, which is always included in models.

    +------------------+---------------------------+
    |    Sentences     |           Terms           |
    +------------------+---------------------------+
    |     VOWEL(A)     |    A, B, 2 - constants    |
    +------------------+---------------------------+
    |   ABOVE(A, B)    |    x, y, z - variables    |
    +------------------+---------------------------+
    |      2 = 2       |  NUMBEROF(A) - functions  |
    +------------------+---------------------------+

Sentences can be combined with all the operators from propositional logic:-

    [; \wedge \vee \lnot \Rightarrow \Leftrightarrow ( ) ;]

We also have 'quantifiers' (which makes first-order logic unique):-

    [; \forall x ;] - For all x
    [; \exists y ;] - There exists a y

E.g. valid sentences in first-order logic:-

    [; \forall x $VOWEL(x)$ \Rightarrow $NUMBEROF(X)$ = 1 ;]

Reads as 'for all x, if x is a vowel then the number of X is equal to 1'.

    [; \exists x $NUMBEROF(x)$ = 2 ;]

Reads as 'there exists an x such that the number of X equals 2'.

Sometimes there's an abbreviation where we omit the quantifier, which means we can simply assume 'for all'.

Typically, when we have a for all quantifier, then it tends to go with a conditional. This is
because we rarely want to say something about literally *every* object, rather we want to say
something about a particular type of object.

Contrary-wise, when we have an 'exists' quantifier, then it tends to be without a conditional
because we're then only talking about a single object.

## Vacuum World ##

Looking at the long-ago mentioned two-dimensional vacuum world:-

<img src="http://codegrunt.co.uk/images/ai/7-vacuum-world-1.png" />

Let's represent it in first-order logic.

Let's define some constants:-

* A - left location
* B - right location
* V - Vacuum
* D1 - Dirt on left
* D2 - Dirt on right

Also some relations:-

* Loc - true of any location
* Vacuum - true of vacuum
* Dirt - true of dirt
* At(o, l) where o is object and l is location.

E.g. if we want to say the vacuum is in A we can say At(V, A).

If we want to say that there is no dirt in any location, then it's a little more involved:-

    [; \forall d \forall l $Dirt(d)$ \wedge $Loc(l)$ \Rightarrow \lnot $At(d, l)$ ;]

What's really useful in first-order logic is that even if there we thousands of locations, this
sentence would still hold.

If we want to say the vacuum is in a location with dirt:-

    [; \exists l \exists d $Dirt(d)$ \wedge $Loc(l)$ \wedge $At(V, l)$ \wedge $At(d, l)$ ;]

What does first-order mean? This means that the relations are on objects rather than relations. If
there were relations on relations, then this is referred to as 'higher-order logic', e.g.:-

    [; \forall R $Transitive(R)$ \Leftrightarrow (\forall a, b, c R(A, b) \wedge R(b, c) \Rightarrow R(a, c)) ;]

This is valid in higher-order logic, but invalid in first-order logic.

## Question 1 + 2 ##

Consider:-

    [; \exists x, y x = y ;]

Valid. All models must have at least one object, and we can have x and y refer to the same
object.

    [; (\exists x x=x) \Rightarrow (\forall y \exists z y=z) ;]

Valid. The left-hand side is true by definition, since x always equals x. As for the right-hand side, we
can always choose a z which is equal to y, so this is also always true.

    [; \forall x P(x) \vee \lnot P(x) ;]

Valid. Either everything is in the relation for P or not.

    [; \exists x P(x) ;]

Satisfiable. P might be an empty relation, for example, and thus this can sometimes be untrue,
however it might also sometimes be true.

Let's look at questions in english and their equivalent representation in first-order logic, where
some of these will be incorrect:-

'Sam has two jobs'

    [; \exists x, y Job(Sam, x) \wedge Job(Sam, y) \wedge \lnot (x=y) ;]

This is a good representation. The final term is critical, since x could equal y.

'Set membership'

Assume the notion of adding an element to a set is defined.

    [; \forall x, s Member(x, Add(x, s)) ;]
    [; \forall x, s Member(x, s) \Rightarrow (\forall y Member(x, Add(y, s))) ;]

This is not a good representation. It is good for indicating that an object is a member of a set,
since we are indicating that adding a single item to a set, means x is a member of the set, also if
x is a member of a set, then it is also a member of a set with another number added, however it does
nothing to indiciate whether a member is *not* in a set, e.g. we'd want to know that '3' wasn't in
the empty set.

'Adjacent squares on a chequers board numbered with x & y coordinates'

    [; \forall x, y Adj(Sq(x, y), Sq(+(x, 1), y)) \wedge
                    Adj(Sq(x, y), Sq(x, +(y, 1))) ;]

This not a good representation. It's good so far in that it tells you that, say, Sq(1, 1) is
adjacent to Sq(2, 1), however it does not tell you that Sq(2, 1) is adjacent to Sq(1, 1), nor what
squares it is *not* adjacent to!

The moral is - you're better off using the iff operator, e.g.:-

    [; \Leftrightarrow ;]
