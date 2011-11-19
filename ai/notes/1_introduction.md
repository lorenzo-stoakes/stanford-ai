AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 1 - Introduction
---------------------

## Course Overview ##

### Purpose of this class ###

* To teach you the basics of artificial intelligence

* To excite you

### Structure ###

Videos -> Quizzes -> Answer Videos

Assignments = quizzes without the answers :-) - graded.

Also, exams.

An AI program is called an 'intelligent agent'.

## Intelligent Agents ##

    Agent
    +----------------+   <- Sensors      Environment
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment
    |                |   Actuators ->    Environment
    +----------------+                   Environment

Happens in many iterations over the perception/action cycle.

## Applications of AI ##

### AI in Finance ###

    Trading Agent
    +----------------+   <- Rates/News   Environment  E.g. stock market
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment  Bonds
    |                |                   Environment
    |                |                   Environment  Commodities
    |                |   Trades ->       Environment
    +----------------+                   Environment

### AI in Robotics ###

    +----------------+   <- Cameras      Environment
    |                |      Microophone  Environment
    |                |      Touch        Environment
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment
    |                |     Motors  ->    Environment
    +----------------+     Voice         Environment

Initial web crawlers were called 'robot's, e.g. robots.txt.

We're going to focus to a large degree on robotics in the course.

### AI in Games ###

    Game Agent
    +----------------+   <- Your moves -  Environment
    |                |                    Environment
    |             |  |                    Environment
    |             |  |                    Environment E.g. you
    |             |  |                    Environment
    |             v  |                    Environment
    |                | - Its own moves -> Environment
    +----------------+                    Environment

### AI in Medicine ###

    Diagnostic Agent
    +----------------+    vital
    |                | <------------ you
    |                |   signals
    |                |
    |                |
    |                | diagnostics
    |                | ------------> doctor
    +----------------+

### AI and the Web ###

I.e. searching for 'great AI class'

    Crawler
    +----------------+   <- Web Pages
    |                |
    | DB             |                   World-wide-web
    |                |
    |                |
    |                |
    |                |
    +----------------+
       ^        |
       | Query  v Results

    <<<<----You---->>>>

## Terminology ##

* Fully- vs. Partially- Observable

Fully = what the agent can sense in the environment is *completely* sufficient for it to make an
optimal decision. E.g. card games where all the cards are the table + visible cards are sufficient
to make decisions.

Partially = Have to maintain a memory, e.g. poker. So not entirely sufficient.

    Agent
    +----------------+   <- Sensors      Environment  ---> state
    |                |                   Environment         |
    |                |                   Environment <--------
    |                |                   Environment
    |                |                   Environment
    |                |                   Environment
    |                |   Actuators ->    Environment
    +----------------+                   Environment

Happens in many iterations over the perception/action cycle.

Fully observable if the sensors can see the entire state of the environment.
Partially observable if the sensors can't fully see state, keeps memory of previous state.

* Deterministic vs. stochastic environment

Deterministic where agent's actions uniquely determine the actions.

E.g. chess - outcome always the same for a given move.

Games like a dice-based game are stochastic - certain amount of randomness involved.

* Discrete vs. continuous

Discrete - Finitely (?! Can't understand what he said...) many action choices, and finitely many
things you can sense, e.g. chess - finitely many board positions, finitely many things you can do.

Continuous - Space of possible actions or things you can sense are infinite, e.g. throwing darts.

* Benign vs. adversarial (environments)

In benign environments, the environment might be random/stochastic, but it has no objective of its
own to contradict your own objectives. E.g. weather - it might be random, but it's not 'out to get
you'.

Adversarial, e.g. chess - where the opponent is actually 'out to get you'. It's much harder to find
good actions. In an adversarial situation, there is an agent which actively observes you +
counteracts what you are trying to achieve.

E.g. checkers - fully-observable, deterministic, discrete + adversarial

## Poker Question ##

E.g. poker - partially-observable, stochastic, discrete + adversarial

## Robot Car Question ##

E.g. robot car - partially-observable, stochastic, continuous, adversarial

Not adversarial apparently :) - quite subjective!

## AI and Uncertainty ##

AI as uncertainty management

AI = what to do when you don't know what to do?

Reasons for uncertainty:-

* Sensor limits
* Adversaries
* Stochastic environments
* Laziness
* Ignorance

## Machine Translation ##

E.g. google translate.

All done via AI/machine learning.

Done by taking lots of data, i.e. where there are already translations out there in language x and
language y.

If people ask for actual translation of article kept in records, then can just go and look that up ('we already have that').

Google also allow translation of individual words by keeping vast amounts of records of words in
different languages.

E.g. Chinese menu

Comparing rows with words common between them, e.g. 'wonton'. Also, the rows with 'wonton' in the
item name are also the only ones which contain a particular Chinese character - makes it highly
likely to be the character in question.

Looked up chicken - common char between Chicken entries.

## Chinese Translation 1 ##

Looking up a phrase - 2 chars common between two entries - 'corn cream'.

## Chinese Translation 2 ##

Lookup up a word again - soup - which os common between entries with 'soup' in them, well apart from
where it's a different character (?!). Correspondence doesn't have to be 100% for there to be a
correlation.

## Summary ##

Completed unit 1

* Key applications of AI
* Intelligent agent
* 4 key attributes
* Sources + management of uncertainty
* Rationality
