AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 3 - Probability in AI
--------------------------

## Introduction ##

We're going to be looking at probabilities, especially structured probabilities, using Bayes
networks. The material is hard.

E.g. Bayes Network - car won't start, causes?

<img src="http://codegrunt.co.uk/images/ai/3-introduction-1.png" />

This is a Bayes network - composed of nodes which correspond to events that you might/might not know
typically called 'random variables', linked by arcs where an arc indicates that the child is
influenced by its parent. Perhaps not in a deterministic way, can be linked in a probabilistic way,
e.g. battery age has a high chance of causing a dead battery, however it's not entirely necessary
(not all old batteries are dead).

There are 16 variables here. What the graph structure + associated probabilities imply is a huge
probability distribution. If we assume they are binary (as we will during this unit), then that's
2^16 possible values.

Once the network is set up, we can observe things like whether the lights are on, or whether the oil
light is on, and compute probabilities for the hypothesis, e.g. alternator broken, etc.

Going to look at how to construct these networks

* Binary events
* Probability
* Simple Bayes Networks
* Conditional Independence
* Bayes Networks
* D-Separation
* Parameter Counts
* Later: Inference in Bayes Networks

Very important.

Bayes used in almost all fields of smart computing, e.g. diagnostics, prediction, machine learning,
finance, google, robotics, particle filters HMM, MDP + POMDPs, Kalman Filters, etc. (we'll find out
about these odd-sounding applications later :-)

## Probability/Coin Flip + 2-5 ##

Cornerstone of AI. Used to express uncertainty, and the management of uncertainty is key in AI.

E.g. flipping a coin:-

    [; P(H) = 0.5, P(T) = 1 - 0.5 = 0.5 ;]

    [; P(H) = 0.25, P(T) = 1 - 0.25 = 0.75 ;]

    [; P(H, H, H) = 1/8 given P(H) = 0.5 ;]

Given:-

    [; x_i = ;] result of i-th coin flip, where [; x_i = { H, T } ;]

Then:-

    [; P(X_1 = X_2 = X_3 = X_4) = 1/(2^4) + 1/(2^4) = 1/8 ;]

Probability of 3 or more heads in 4 flips:-

    [; P({X_1, X_2, X_3, X_4} contains /geq 3 H) = 5/16 = 0.3125 ;]

Examined all the possibile outcomes shows that there are 5 possibilities:-

    HHHH *
    HHHT *
    HHTH *
    HHTT
    HTHH *
    HTHT
    HTTH
    HTTT
    THHH *
    THHT
    THTH
    THTT
    TTHH
    TTHT
    TTTH
    TTTT

## Probability Summary ##

    [; P(A) = p => P(notA) = 1 - p ;]

Independence:-

    [; X \perp Y: P(X)P(Y)   = P(X, Y) ;]
                 marginals    joint probability

## Dependence ##

<img src="http://codegrunt.co.uk/images/ai/3-dependence-1.png" />

## What We Learned ##

### Lessons ###

    [; P(Y) = \sum_i(P(Y|X=i)P(X=i)) ;]

    [; P(\lnot X | Y) = 1 - P(X | Y) ;]

    [; P(X | \lnot Y) = 1 - P(X | Y) ;] <- NOT TRUE!

## Weather + 2-3 ##

    [; P(D_1 = sunny) = 0.9 ;]
    [; P(D_2 = sunny | D_1 = sunny) = 0.8 ;]
    [; P(D_2 = sunny | D_1 = rainy) = 0.6 ;]

Since

    [; P(notX | Y) = 1 - P(X | Y) ;] :-
    [; P(D_2 = rainy | D_1 = sunny) = 1 - P(D_2 = sunny | D_1 = sunny) = 0.2 ;]

Similarly:-

    [; P(D_2 = rainy | D_1 = rainy) = 1 - 0.6 = 0.4 ;]

By the Theory of Total Probability:-

    [; P(D_2 = sunny) = P(D_2 = sunny | D_1 = sunny)P(D_1 = sunny) + P(D_2 = sunny | D_1 = rainy)P(D_1 = rainy) ;]

And:-

    [; P(D_3 = sunny) = P(D_3 = sunny | D_2 = sunny)P(D_2 = sunny) + P(D_3 = sunny | D_2 = rainy)P(D_2 = rainy) ;]

Note that we assume the conditional probabilities are the same for D2->D3 as they are for D1->D2.

So:-

    [; P(D_2 = sunny) = 0.8 \times 0.9 + 0.6 \times (1 - 0.9) = 0.8 \times 0.9 + 0.6 \times 0.1 = 0.78 ;]
    [; P(D_3 = sunny) = 0.8 \times 0.78 + 0.6 \times (1 - 0.78) = 0.8 \times 0.78 + 0.6 \times 0.22 = 0.756 ;]

## Cancer + 2-4 ##

We can express the probability of cancer/not cancer thus;-

    [; P(C) = 0.01 ;]
    [; P(\lnot C) = 1 - 0.01 = 0.99 ;]

Let's say there's a test which comes out positive (+) or negative (-):-

    [; P(+|C) = 0.9 ;]
    [; P(-|C) = 1 - 0.9 = 0.1 ;]

Since

    [; p(\lnot A|B) = 1 - P(A|B) ;]

    [; P(+|\lnot C) = 0.2 ;]
    [; P(-|\lnot C) = 0.8 ;]

We're after:-

    [; P(C|+) ;]

But, first let's get some joint probabilities:-

<img src="http://codegrunt.co.uk/images/ai/3-cancer-1.png" />

Due to the rule of total probability:-

    [; P(+) = P(+|C)P(C) + P(+|\lnot C)P(\lnot C) = 0.9 \times 0.01 + 0.2 \times 0.99 = 0.207 ;]

We applied a sort of ad-hoc method to obtain the joint probabilities which, more formally, results
in:-

    [; P(A, B) = P(A)P(B|A) = P(B)P(A|B) ;]

So:-

    [; P(B|A) = \frac{P(A, B)}{P(A)} ;]
    [; P(A|B) = \frac{P(A, B)}{P(B)} ;]

So
    [; P(C|+) = \frac{P(C, +)}{P(+)} = \frac{0.009}{0.207} \simeq 0.0435 ;]

## Bayes Rule ##

Invented by Rev. Thomas Bayes, mathematician.

    [; P(A|B) = \frac{P(B|A)P(A)}{P(B)} ;]

Where:-

    [; P(A|B) $ is the posterior$ ;]
    [; P(B|A) $ is the likelihood$ ;]
    [; P(A) $ is the prior$ ;]
    [; P(B) $ is the marginal likelihood$ ;]

Let's say that B is the evidence, and A is what we're interested in, e.g. test result vs cancer.

This is 'diagnostic reasoning' - given evidence, looking at the cause, i.e. - given something
observable, what is the probability of the non-observable thing?

Bayes turns this upside down to 'causal reasoning':-

>'Given, hypothetically, we knew the cause, what would be the probability of the evidence we just
> observed?'

To correct for this inversion, we have to multiply by the prior of the cause to be the case in the
first place, and divide it by the probability of the evidence, which is often expanded to:-

    [; P(A|B) = \frac{P(B|A)P(A)}{\sum_a P(B|A=a)P(A=a)} ;]

Using total probability.

    [; P(C|+) = \frac{P(+|C)P(C)}{P(+)} = \frac{P(+|C)P(C)}{P(+|C)P(C) + P(+|C)P(\lnot C)} ;]
    [; = 0.9 * 0.01 / (0.9 * 0.01 + 0.2 * 0.99) ;]
    [; \simeq 0.0435 ;]

Which is the same as above :)

## Bayes Network ##

We can represent the kind of reasoning we performed in the above example graphically:-

<img src="http://codegrunt.co.uk/images/ai/3-bayes-network-1.png" />

The information on the right-hand column is what we have, and we want to perform diagnostic
reasoning, i.e. determining P(A|B) and P(A|notB).

Quiz - how many parameters:-

<img src="http://codegrunt.co.uk/images/ai/3-bayes-network-2.png" />

## Computing Bayes Rule ##

Looking at more complex networks.

Examining Bayes Rule again:-

    [; P(A|B) = \frac{P(B|A)P(A)}{P(B)} ;]

P(B|A) and P(A) are relatively easy to determine. P(B), not so much.

However, we can take a look at the negation of P(A|B), and cancel this term out:-

    [; P(\lnot A|B) = \frac{P(B|\lnot A)P(\lnot A)}{P(B)} ;]

We can simply ignore the 'normaliser' P(B), to give us P' 'psuedo-probability' terms:-

    [; P'(A|B) = P(B|A)P(A) ;]
    [; P'(\lnot A|B) = P(B|\lnot A)P(\lnot A) ;]

If we define a normaliser, eta, then we can get back to the actual probabilities thus:-

    [; P(A|B) = \eta P'(A|B) ;]
    [; P(\lnot A|B) = \eta P'(\lnot A|B) ;]

Where:-

    [; \eta = (P'(A|B) + P'(\lnot A|B))^{-1} ;]

We normalise the pseudo probabilities such that total probability holds:-

    [; P(A|B) + P(\lnot A|B) = 1 ;]

## Two Test Cancer + 2 ##

<img src="http://codegrunt.co.uk/images/ai/3-two-test-cancer-1.png" />

We declare a short-form:-

    [; P(C|T_1 = + T_2 = +) = P(C|++) ;]

So:-

    [; P(C|++) = \eta P'(C|++) ;]

And:-

    [; P'(C|+) = P(+|C)P(C) ;]

Assuming each test is independent of one another (conditionally independent, see below), then we
simply multiply by another P(+|C):-

Note: This seems extremely hand-wavy to me.

    [; P'(C|++) = P(+|C)P(+|C)P(C) = 0.9 \times 0.9 \times 0.01 = 0.0081 ;]
    [; P'(\lnot C|++) = P(+|\lnot C)P(+|\lnot C)P(\lnot C) = 0.2 \times 0.2 \times 0.99 = 0.0396 ;]

And to normalise:-

    [; \eta = (P'(C|++) + P'(\lnot C|++))^{-1} = 0.0081 \times 0.0396 \simeq 20.96 ;]

Thus:-

    [; P(C|++) = 20.96 \times 0.081 \simeq 0.1698 ;]
    [; P(\lnot C|++) = 20.96 \times 0.0396 \simeq 0.8302 ;]

Again, to determine P(C|+-), we follow a similar procedure:-

    [; P(C|+-) = \eta P'(C|+-) ;]
    [; P'(C|+-) = P(+|C)P(-|C)P(C) = 0.9 \times 0.1 \times 0.01 = 0.0009 ;]
    [; P'(\lnot C|+-) = P(+|\lnot C)P(-|\lnot C)P(\lnot C) = 0.2 \times 0.8 \times 0.99 = 0.1584 ;]
    [; \eta = (P'(C|+-) + P'(\lnot C|+-))^{-1} = (0.0009 + 0.1584)^{-1} \simeq 6.277 ;]

Thus:-

    [; P(C|+-) = 6.277 \times 0.0009 \simeq 0.00565 ;]
    [; P(\lnot C|+-) = 6.277 \times 0.1584 \simeq 0.994 ;]

## Conditional Independence + 2 ##

Introducing some terminology.

The 'hidden variable' C causes the stochastic test outcomes T1 and T2.

We didn't just assume that T1 and T2 are identically distributed, we also assumed that they were
*conditionally independent*.

If we knew with absolute certainty the value of C, it would tell us nothing to relate T1 to T2.

More formally:-

    [; P(T_2|C, T_1) = P(T_2|C) ;]

I.e. knowledge of T1 has absolutely no impact on T2.

This follows from the Bayes diagram - if we removed C from it, then T1 and T2 are essentially cut
off from one another.

Conditional independence is really important.

Looking at the following more general diagram:-

<img src="http://codegrunt.co.uk/images/ai/3-conditional-independence-1.png" />

Then we write:-

    [; Given A, B \perp C ;]

    [; B \perp C | A ;]

If we *don't* know the prior condition, i.e. A, then we *can't* say that B and C are
independent. This is because the result of B indicates something about the hidden A which will in
turn influence C. In our cancer case, one positive test result gives an indication as to whether we
have cancer, and thus another test result will be influenced by that too.

To drive the point home - let's calculate the probability of one test given the result of another:-

Let's use some short-hand once again:-

    [; P(T_1=+) = P(+_1), P(T_2=+) = P(+_2) ;]

We can use total probability to answer this:-

    [; P(+_2|+_1) = P(+_2|+_1, C)P(C|+_1) + P(+_2|+_1, \lnot C)P(\lnot C|+_1) ;]

Given conditional independence, this is equivalent to:-

    [; P(+_2|+_1) = P(+_2|C)P(C|+_1) + P(+_2|\lnot C)P(\lnot C|+_1) ;]

    [; P(+_2|+_1) = 0.9 \times 0.0435 + 0.2 \times 0.9565 \simeq 0.2305 ;]

## Absolute and Conditional ##

Let's look at the different forms of independence we've encountered:-

<img src="http://codegrunt.co.uk/images/ai/3-absolute-and-conditional-1.png" />

    [; A \perp B \not\Rightarrow A \perp B | C ;]

We will discuss why this is the case next :-)

    [; A \perp B | C \not\Rightarrow A \perp B ;]

As we've seen just now, conditional independence does not imply absolute independence since
something which affects one conditionally independent event can also affect the other.

## Confounding Cause ##

<img src="http://codegrunt.co.uk/images/ai/3-confounding-cause-1.png" />

    [; P(S)=0.7 ;]
    [; P(R)=0.01 ;]

    [; P(H|S, R) = 1 ;]
    [; P(H|\lnot S, R) = 0.9 ;]
    [; P(H|S, \lnot R) = 0.7 ;]
    [; P(H|\lnot S, \lnot R) = 0.1 ;]

This is a trick question. Since P(R) and P(S) are independent, P(R|S) = P(R) = 0.01!

## Explaining Away + 2-3 ##

(Again, working with the example given above)

Explaining away means - if we know that we are happy, then sunny weather can 'explain away' the
cause of happiness. If it's sunny, then it makes it less likely that there has been a raise.

If it's rainy, then it makes it more likely to be a raise since the happiness cannot be explained by
the weather.

If we see a certain effect which can be caused by multiple causes, then seeing one of those causes
can 'explain away' any other cause.

E.g., we want to determine:-

    [; P(R|H, S) ;]

We can use a sneaky trick here, by using a twist on Bayes:-

    [; P(A|B,C) = \frac{P(B|A,C)P(A|C)}{P(B|C)} ;]

I.e. - we still do the switch between A and B, only we take into account the fact that everything is
still predicated on C.

So:-

    [; P(R|H, S) = \frac{P(H|R, S)P(R|S)}{P(H|S)} = \frac{0.01}{P(H|S)};]

Since

    [; P(H|R, S) = 1 ;]
    [; P(R|S) = P(R) = 0.01 ;]

Carrying on:-

    [; P(H|S) = P(H|R, S)P(R|S) + P(H|\lnot R, S)P(\lnot R|S) ;]
    [; P(H|S) = 0.01 + 0.7 \times 0.99 = 0.703;]

Hence:-

    [; P(R|H, S) = \frac{0.01}{0.703} \simeq 0.0142 ;]

Let's determine P(R|H):-

    [; P(R|H) = \eta P'(R|H) ;]
    [; P'(R|H) = P(H|R)P(R) ;]
    [; P'(\lnot R|H) = P(H|\lnot R)P(\lnot R) ;]

    [; P(H|R) = P(H|R,S)P(S|R) + P(H|R,\lnot S)P(\lnot S|R) ;]

Again, since S and R are fully independent:-

    [; P(H|R) = P(H|R,S)P(S) + P(H|R,\lnot S)P(\lnot S) ;]
    [; P(H|R) = 1 \times 0.7 + 0.9 \times 0.3 = 0.97 ;]

Going through the same process for [; P(H|\lnot R) ;]:-

    [; P(H|\lnot R) = P(H|\lnot R, S)P(S|\lnot R) + P(H|\lnot R, \lnot S)P(\lnot S|\lnot R) ;]
    [; P(H|\lnot R) = 0.7 \times 0.7 + 0.1 \times 0.3 = 0.52 ;]

So now we can calculate the pseudo probabilities:-

    [; P'(R|H) = 0.97 \times 0.01 = 0.0097 ;]
    [; P'(\lnot R|H) = 0.52 \times 0.99 = 0.5148 ;]

And the normalisation factor:-

    [; \eta = (P'(R|H) + P'(\lnot R|H))^{-1} ;]
    [; \eta = (0.0097 + 0.5148)^{-1} \simeq 1.9066 ;]

Hence:-

    [; P(R|H) = 1.9066 \times 0.0097 \simeq 0.0185 ;]

The point here is that if he's happy but doesn't know about the weather, then the probability of a
raise is higher. The knowledge of the weather reduces the probability of the raise.

To calculate [; P(R|H, \lnot S) ;], we can (ab)use Bayes again:-

    [; P(R|H, \lnot S) = \frac{P(H|R, \lnot S)P(R|\lnot S)}{P(H|\lnot S)};]

And using the theorem of total probability one more:-

    [; P(H|\lnot S) = P(H| \lnot S, R)P(R|\lnot S) + P(H|\lnot S, \lnot R)P(\lnot R|\lnot S) ;]

Again, S and R are totally independent so:-

    [; P(H|\lnot S) = P(H| \lnot S, R)P(R) + P(H| \lnot S, \lnot R)P(\lnot R) ;]
    [; P(H|\lnot S) = 0.9 \times 0.01 + 0.1 \times 0.99 = 0.108;]

Also, let's take this into account for our original equation:-

    [; P(R|H, \lnot S) = \frac{P(H|R, \lnot S)P(R)}{P(H|\lnot S)} ;]

So, finally:-

    [; P(R|H, \lnot S) = \frac{0.9 \times 0.01}{0.108} \simeq 0.0833 ;]

## Conditional Dependence ##

It's interesting to compare all the outcomes regarding the raise:-

    [; P(R|S) = 0.01 ;]
    [; P(R|H, S) = 0.0142 ;]
    [; P(R|H, \lnot S) = 0.0833 ;]

H adds a dependence between S and R, despite them being independent.

<img src="http://codegrunt.co.uk/images/ai/3-conditional-dependence-1.png" />

Without information about H, the probability of R is completely unaffected by the knowledge of H.

    [; R \perp S ;]

However, when we know something about H, then things begin to get affected, i.e.:-

    [; P(R|H, S) = 0.0142 \not= P(R|H) ;]
    [; P(R|S) = 0.01 = P(R) ;]
    [; P(R|H, \lnot S) = 0.0833 \not= P(R|H) ;]

The probability of a raise, R, is affected by the probability of sunny weather.

This leads to the previously mentioned fact that full independence does not mean conditional
independence, i.e.:-

    [; R \perp S ;]
    [; R \not\perp S | H ;]

So, two variables that are independent might not be conditionally independent.

INDEPENDENCE DOES __NOT__ IMPLY CONDITIONAL INDEPENDENCE!

## General Bayes Net + 2-3 ##

We can now define Bayes networks in a more general way. Bayes networks define probability
distributions over a graph of random variables, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/3-general-bayes-net-1.png" />

Instead of enumerating all possibilities of all combinations of these 5 random variables, the Bayes
network is defined by probability distributions which is inherent to each individual node.

The probability at each node is only conditioned on the incoming arcs.

* A has no incoming arcs, so its probability is P(A).
* B has no incoming arcs, so its probability is P(B).
* C has incoming arcs from A and B, so its probability is P(C|A, B).
* D has an incoming arc from C, so its probability is P(D|C).
* E has an incoming artc from C, so its probability is P(E|C).

Thus:-

    [; P(A, B, C, D, E) = P(A).P(B).P(C|A, B).P(D|C).P(E|C) ;]

This has a big advantage in that the joint distribution of any five variables requires 2^5-1 = 31
probability values, whereas the Bayes network requires only 10, e.g.:-

2 From P(A), P(B):-

    [; P(A), P(B) =2 ;]

4 from P(C|A, B):-

    [; P(C|A, B), P(C|A, \lnot B), P(C|\lnot A, B), P(C|\lnot A, \lnot B) ;]

2 from P(D|C):-

    [; P(D|C), P(D|\lnot C) ;]

2 from P(E|C):-

    [; P(E|C), P(E|\lnot C) ;]

Scales a lot better to large networks than the combinatorial approach.

Some quizzes:-

<img src="http://codegrunt.co.uk/images/ai/3-general-bayes-net-2.png" />

<img src="http://codegrunt.co.uk/images/ai/3-general-bayes-net-3.png" />

## Value of a Network ##

And our original network:-

<img src="http://codegrunt.co.uk/images/ai/3-introduction-1.png" />

Which is:-

Row 1:-

    [; 1 + 1 + 1 = 3 ;]

Row 2:-

    [; 2 + 2^2 = 6 ;]

Row 3:-

    [; 2 + 2^2 + 1 + 1 + 1 + 1 = 10 ;]

Row 4:-

    [; 2 + 2^2 + 2^2 + 2^4 + 2 = 28 ;]

So:-

Total =

    [; 3 + 6 + 10 + 28 = 47 ;]

Which is quite an improvement on 65,535 using the combinatorial approach!

## D-Separation + 2-3 ##

<img src="http://codegrunt.co.uk/images/ai/3-d-separation-1.png" />

So:-

    [; C \perp A ;]

No, since A has an effect on C.

    [; C \perp A|B ;]

Yes, since once we assume B, we've taken into account A's effect.

    [; C \perp D ;]

No, since A effects C, which also effects D.

    [; C \perp D|A ;]

Yes, since A is the 'common ancestor' between C and D.

    [; E \perp C|D ;]

Yes, since D takes into account A's effect which is the common ancestor between C and E.

Put simply:-

Any two nodes are independent if they're not linked by just unknown variables. E.g., if we know B,
then anything downstream of B is independent of everything upstream of B.

<img src="http://codegrunt.co.uk/images/ai/3-d-separation-2.png" />

    [; A \perp E ;]

No, since A influences C which influences E.

    [; A \perp E | B ;]

No, since B doesn't exclude A's influence from affecting E.

    [; A \perp E | C ;]

Yes, since C does exclude A's influence on E.

    [; A \perp B ;]

Yup.

    [; A \perp B | C ;]

No, since C can explain things away.

This leads to the general study of conditional independence in Bayes networks, often calle D-separation or reachability.

D-separation is best studied by 'active tripets' and 'inactive triplets'.

* Active triplets render variables dependent.
* Inactive triplets render variables independent.

E.g.:-

<img src="http://codegrunt.co.uk/images/ai/3-d-separation-3.png" />

Final Quiz :-) :-

<img src="http://codegrunt.co.uk/images/ai/3-d-separation-4.png" />

Considering independence:-

    [; F \perp A ;]

Yes.

    [; F \perp A | D ;]

No, as D helps 'explain away' B and E, which then percolates up to A and F, hence they are *not*
independent.

    [; F \perp A | G ;]

Again, knowledge of G percolates up to do, which then 'explains away' B and E, which go back to A
and F, hence A and F are *not* independent.

    [; F \perp A | H ;]

H might tell us something about G, but it won't tell us anything about D, so A and F are not independent.

## Congratulations! ##

Learnt a lot:-

* Graph structure
* Compact Representation
* Conditional Independence

This was a largely theoretical unit. Will talk more about applications later.

Bayes networks are very useful for many applications.
