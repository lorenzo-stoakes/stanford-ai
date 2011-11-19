AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 5 - Machine Learning
-------------------------

## Introduction ##

* Lots of data in the world e.g. dna, finance, web, etc.

Machine learning is the discipline of extracting information from this data.

## What is Machine Learning ##

We've already talked about Bayes networks.
Machine learning is about finding these networks based on data.
Machine learning = learn models from data.

E.g. google/amazon/etc.

We'll start with supervised learning then moving on to unsupervised learning.

## Stanley DARPA Grand Challenge ##

Stanley is a self-driving car - no human assistance whatsoever.

Uses a laser system to build models of the terrain. Problem is, they only see 25m. If you drive fast
you have to see further.

Uses machine learning principles on the laser-scanned portion of the road in conjunction with the
camera view in order to see right up to the horizon.

Means you can drive fast.

Key factor in winning the race.

## Taxonomy ##

Very large field

Very basic terminology:-

### What? ###

* Parameters - e.g. probabilities of a Bayes network.
* Structure - e.g. Arc structure of a Bayes network.
* Hidden concepts - e.g. Certain training example might form a group. Can help you make better sense
  of the data.

### What From? ###

Machine learning driven by some sort of information that you care about:-

* Supervised learning - We use specific target labels.
* Unsupervised learning - target labels are missing and we use replacement principles to find, for
  example, hidden concepts.
* Reinforcement learning - An agent learns from feedback from a physical environment, where it
  receives some sort of response to actions, e.g. 'well done' or 'that works'.

### What For? ###

* Prediction - e.g. stock market
* Diagnostics - e.g. medicine
* Summarisation - e.g. summarising a long article

### How ###

* Passive - If the agent has no impact on the data itself.
* Active - The agent has some impact.
* Online - Data obtained while being generated.
* Offline - Data obtained after been generated.

### Outputs? ###

* Classification - Output is binary/fixed number of classes e.g. chair or not
* Regression - Output is continuous, e.g. 66.5 degrees Celcius

### Details? ###

* Generative - Seek to model the data as generally as possible.
* Discriminative - Seek to distinguish data.

Might seem like a trivial distinction, however it makes a huge difference in the implementation of
the algorithm.

### Supervised Learning ###

Most of the work in the field is supervised learning.

For each training example given a feature vector and a target label called y:-

    [; x_1 x_2 x_3 ... x_n \to y ;]

e.g. for credit rating - feature vector = person's salary, whether they've defaulted before, etc.,
target label = credit rating.

Based on previous data and actual instances of default. The idea is to predict future customer
outcomes, e.g. person comes in with a different feature vector - are they likely to default or not?

Can apply the exact same idea to image recognition, e.g. x's are pixels, and y is whether or not recognised:-

    [; \begin{bmatrix}
        x_{11} & x_{12} & x_{13} & ... & x_{1n} & \to & y_1 \\
        x_{21} & x_{22} & x_{23} & ... & x_{2n} & \to & y_2 \\
        ...    & ...    & ...    & ... & ...    & ... & ... \\
        x_{m1} & x_{m2} & x_{m3} & ... & x_{mn} & \to & y_m \\
       \end{bmatrix} ;]

We call this the data. For each vector

    [; x_m ;]

We want to find a function as follows:-

    [; f(x_m) = y_m ;]

This isn't always possible, so sometimes it's tolerable to permit a certain amount of error.

We can then use this function for future x's to obtain accurate y's.

## Occam's Razor ##

<img src="http://codegrunt.co.uk/images/ai/5-occams-razor-1.png" />

If you compare a. to b., you can see that both fit the data perfectly, however b is clearly less
appropriate as it violently oscillates all over the place and it is unlikely (let's face it -
impossible) that it will actually fit the data between these points.

> Everything else being equal, choose the less complex hypothesis

There is a trade-off:-

    Fit <--------------------> Low Complexity

E.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-occams-razor-2.png" />

Here complexity could be polynomial degree. As complexity increases, training data error reduces as
you fit the data more accurately, however overall error (generalisation error) increases, which is
the sum of training data error and 'overfitting' error.

The best complexity is obtained is where the generalisation error is at a minimum.

There are methods for determining the overfitting error, in a field called 'Bayes variance methods'.

Often in practice you only have the training data error. It turns out that if you don't fully
minimise training data error but instead push back on the complexity, your algorithm tends to
perform better.

If you deal with data + have means of fitting your data, keep in mind overfitting is often a source
of error.

## Spam Detection ##

This is a good application of machine learning.

This is a discrimination problem, can use supervised machine learning for this.

<img src="http://codegrunt.co.uk/images/ai/5-spam-detection-1.png" />

Here we are trying to determine the function which identifies email as either spam or 'ham' (legit
email). Most systems use human input, flagging spam (unflagged emails are considered ham).

Machine learning algorithm with input is email, output is spam?

### Bag of Words ###

How do we represent emails? Needs to take into account varying email contents e.g. images,
encodings, cases, etc.

Often the method used is 'bag of words' where we essentially keep a dictionary of words with word
counts, e.g.:-

    Hello I will say hello

Would become:-

    hello:2, i:1, will:1, say:1

The count is oblivious to the order of words.

We could in theory use a dictionary which doesn't actually include all the words, e.g.:-

    hello: 2, good-bye: 0

However we tend to include all words.

## Question ##

What is the probability of mail falling into spam bucket given all messages chosen at random?

3 spam messages, 5 ham messages, so 3/8 messages are spam = 3/8 probability.

## Relationship to Bayes Networks ##

### Maximum Likelihood ###

Looking at the previous example, more formally. We have the following emails:-

    SSSHHHHH

We want to determine:-

    [; p(S) = \pi ;]

Now:-

    [; P(y_i)=
       \left\{\begin{matrix}
       \pi $ if $ y_i = S\\
       1-\pi $ if $ y_i = H
       \end{matrix}\right. ;]

If instead of denoting spam 'S' and ham 'H' we denote spam 1 and ham 0:-

    11100000

    [; p(y_i) = \pi^{y_i} . (1-\pi)^{1-y_i} ;]

Assuming independence:-

    [; p(data) = \prod_{i=1}^8 p(y_i) = \pi^{count(y_i=1)}.(1-\pi)^{count(y_i=0)} ;]
    [;         = \pi^3.(1-\pi)^5 ;]

We can maximise the log of this, e.g.:-

    [; \log p(data) = 3.\log\pi + 5\log(1-\pi) ;]

The maximum of this is obtained when the derivative is 0, so let's perform the calculation:-

    [; \frac{\mathrm{d\log p(data)} }{\mathrm{d} \pi}=0 ;]
    [; = \frac{3}{\pi} - \frac{5}{1-\pi} ;]

So:-

    [; \frac{3}{\pi} = \frac{5}{1-\pi} ;]
    [; 3(1-\pi) = 5\pi ;]
    [; \pi = \frac{8}{3} ;]

Which is what we obtained previously.

So counting spam emails and dividing by total was mathematically legitimate.

## Relationship to Bayes Network ##

We're building up a Bayes network here, with parameters estimated via supervised learning via
maximum likelihood estimator.

<img src="http://codegrunt.co.uk/images/ai/5-relationship-to-bayes-network-1.png" />

At root it has an unobservable variable called 'spam' which is binary, and has as many children as
there are words in a message, where each word has an identical conditional distribution of the word
occurrence given the class spam or not spam.

In our previous example we have the following words:-

    offer
    is
    secret
    click
    sports

And we determined:-

    [; p("secret"|SPAM) = \frac{1}{3} ;]
    [; p("secret"|HAM) = \frac{1}{15} ;]

How many parameters required for a 12 word network?

Rather than the anticipated 24 + 1 (given our previous parameter count approach), the answer is
actually 22 + 1, since we know the sum of all word probabilities given S is 1, thus can obtain the
last parameter by 1 - (sum of all other probabilities given spam). The same goes for ham, so we can
subtract 2 here.

## Question 1 ##

Message m = "sports"

We want:-

    [; P(spam|m) ;]


Given our sample set:-

    SPAM

    OFFER IS SECRET
    CLICK SECRET LINK
    SECRET SPORTS LINK

    HAM

    PLAY SPORTS TODAY
    WENT PLAY SPORTS
    SECRET SPORTS EVENT
    SPORTS IS TODAY
    SPORTS COSTS MONEY

We can see from this:-

    [; P(m|spam) = \frac{1}{9} ;]
    [; P(m) = \frac{6}{24} = \frac{1}{4} ;]
    [; P(spam) = \frac{3}{8} ;]

So by Bayes' rule:-

    [; P(spam|m) = \frac{\frac{1}{9} \times \frac{3}{8}}{\frac{1}{4}} = \frac{1}{6} ;]


## Question 2 ##

Message m = "secret is secret"

We want:-

    [; P(spam|m) ;]

So, via Bayes we have:-

    [; P(spam|m) = \frac{P(m|spam)P(spam)}{P(m)} ;]
    [; = \frac{P(m|spam)P(spam)}{P(m|spam)P(spam) + P(m|\lnot spam)P(\lnot spam)} ;]

We can simply count the items in each collection to determine that:-

    [; P(spam) = \frac{3}{8} ;]
    [; P(\lnot spam) = \frac{5}{8} ;]

If we assume that probability of words is conditionally independent on spam/not spam and given that
we are representing emails by bag of words, i.e. we don't care about the order, then:-

    [; P(m|spam) = P($secret is secret$|spam) = P(secret|spam) \times P(secret|spam) \times P(is|spam) ;]

And similar for the ham case.

So we have:-

    [; P(m|spam) = \frac{1}{3} \times \frac{1}{3} \times \frac{1}{9} = \frac{1}{81} ;]
    [; P(m|\lnot spam) = \frac{1}{15} \times \frac{1}{15} \times \frac{1}{15} = \frac{1}{3375};]

And:-

    [; P(spam|m) = \frac{P(m|spam)P(spam)}{P(m|spam)P(spam) + P(m|\lnot spam)P(\lnot spam)} ;]

So:-

    [; P(spam|m) = \frac{\frac{1}{81} \times \frac{3}{8}}{\frac{1}{81} \times \frac{3}{8} + \frac{1}{3375} \times \frac{5}{8}} \simeq 0.9615 ;]

## Question 3 ##

M = "today is secret"

We can see straight away that 'today' isn't in the spam set, thus:-

    [; P(today|spam) = 0 ;]

And:-

    [; P(m|spam) = 0 ;]

So:-

    [; P(spam|m) = 0 ;]

## Answer and Laplace Smoothing ##

According to our model 'today is secret' is simply impossible to be spam. This isn't quite what
you'd expect. This is just overfitting. A single word shouldn't determine the outcome of our entire
model.

### LaPlace Smoothing ###

In maximum likelihood calculations, we have:-

    [; p(x) = \frac{count(x)}{N} ;]

In LaPlace smooth we use the following equation:-

    [; p(x) = \frac{count(x) + k}{N + k|x|} ;]

Where:-

* count(x) - number of occurrences of this value of the variable x.
* |x| - number of values that x can take on.
* k - smoothing parameter.
* N - total number of occurrences of x (the variable rather than value) in the sample size.

Various cases:-

1 message, 1 spam:-

    [; p(x) = \frac{1 + 1}{1 + 1 \times 2} = \frac{2}{3} ;]

10 messages, 6 spam:-

    [; p(x) = \frac{6 + 1}{10 + 1 \times 2} = \frac{7}{12} ;]

100 messages, 60 spam:-

    [; p(x) = \frac{60 + 1}{100 + 1 \times 2} = \frac{61}{102} ;]

## Question 1 ##

Assuming vocab. size = 12 and k = 1.

Since:-

    [; p(x) = \frac{count(x) + k}{N + k|x|} ;]

We have:-

    [; P(spam) = \frac{3 + 1}{8 + 1 \times 2} = \frac{2}{5} ;]
    [; P(ham) = P(\lnot spam) = 1 - P(spam) = \frac{3}{5} ;]

Assuming we keep a dictionary of 12 words:-

    [; P("today"|spam) = \frac{0 + 1}{9 + 1 \times 12} = \frac{1}{21} ;]
    [; P("today"|ham) = \frac{2 + 1}{15 + 1 \times 12} = \frac{3}{27} = \frac{1}{9} ;]

## Question 2 ##

M = "today is secret"
k = 1

We want:-

    [; P(spam|m) ;]

From our previous calculations:-

    [; P(spam|m) = \frac{P(m|spam)P(spam)}{P(m|spam)P(spam) + P(m|\lnot spam)P(\lnot spam)} ;]
    [; P(m|spam) = P("today"|spam)P("is"|spam)P("secret"|spam) ;]
    [; P(m|\lnot spam) = P("today"|\lnot spam)P("is"|\lnot spam)P("secret"|\lnot spam) ;]

We can specify the following for the |spam and |ham (i.e. [; \lnot spam ;]) instances:-

    [; P(x=word|spam) = \frac{count(word) + 1}{9 + 1 \times 12} = \frac{count(word) + 1}{21} ;]
    [; P(x=word|\lnot spam) = \frac{count(word) + 1}{15 + 1 \times 12} = \frac{count(word) + 1}{27} ;]

So:-

    [; P("today"|spam) = \frac{1}{21} ;]
    [; P("is"|spam) = \frac{2}{21} ;]
    [; P("secret"|spam) = \frac{4}{21} ;]
    [; P("today"|\lnot spam) = \frac{3}{27} = \frac{1}{9};]
    [; P("is"|\lnot spam) = \frac{2}{27} = ;]
    [; P("secret"|\lnot spam) = \frac{2}{27} ;]

And:-

    [; P(m|spam) = \frac{1}{21} \times \frac{2}{21} \times \frac{4}{21} \simeq 0.000864 ;]
    [; P(m|\lnot spam) = \frac{1}{9} \times \frac{2}{27} \times \frac{2}{27} \simeq 0.000610 ;]

We know from previous calculations that:-

    [; P(spam) = \frac{2}{5} ;]
    [; P(\lnot spam) = \frac{3}{5} ;]

So:-

    [; P(spam|m) = \frac{0.000864 \times \frac{2}{5}}{0.000864 \times \frac{2}{5}  + 0.000610 \times \frac{3}{5}} \simeq 0.486 ;]

## Summary Naive Bayes ##

So we've learnt about 'naive Bayes', where we've used maximum likelihood and a Laplacian smoother to
determine class probabilities via Bayes rule:-

<img src="http://codegrunt.co.uk/images/ai/5-summary-naive-bayes-1.png" />

This is called a 'generative model' in the conditional probabilities all aim to maximise the
predictability of individual features as if they describe the physical world.

We also use a 'bag of words' model which counts words but doesn't take into account order.

Spammers have found ways of circumventing the naive Bayes method so you have to work harder to
counter 'em.

## Advanced SPAM Filtering ##

Features you may consider:-

* From known spamming IP?
* Have you emailed this person before?
* Have 1,000 other people recently received the same message?
* Email header consistent?
* Email all caps?
* Do inline URLs point to where they say they do?
* Are you addressed by name?

Can add them to the naive base model.

## Digit Recognition ##

Naive Bayes can also be used for handwriting recognition, specifically digits.

We could do the following:-

Input vector = pixel values
Say, 16 x 16

Not vary 'shift invariant', e.g. can't recognise that the right figure below is just the left figure
shifted to the right:-

<img src="http://codegrunt.co.uk/images/ai/5-digit-recognition-1.png" />

There's many different solutions, one of which is to use smoothing, though in a different way than
discussed before. Instead of counting 1 pixel's value count, we can mix it with counts of the
neighbouring pixel values too, so we get similar statistics for a pixel whether shifted or not.

This is called 'input smoothing', technically we 'convolve' the pixels with the Guassian
variable. This might give us better results.

Naive Bayes is not a good choice here, however, since the conditional independence of each pixel is
too strong an assumption to make. However, it's still fun to talk about it :-)

## Overfitting Prevention ##

We've talked about Occam's razor previously - suggests a trade-off between how well we can fit the
data and how 'smooth' our learning algorithm is. We've already seen LaPlacian smoothing as well as
input smoothing.

The question is - how do we choose the smoothing parameter?! There is a method called 'cross
validation'. The method assumes that you have a lot of training data. However this is usually not a
problem for spam.

We begin by dividing our training data into 3 parts:-

* Train - usually around 80% of the data.
* Cross-Validation (CV) - ~ 10%.
* Test - ~ 10%.

You use the train to find all parameters, e.g. probabilities of Bayes network. You then use the cv
set to find an optimal k - you train for different values of k, observe how well the train model
performs on the cv data, leaving the test data then you maximise over all k's to get the best
performance on the cv dataset. You train agan, then test only once against the test data to verify
performance, which is the performance you report.

It's really important to split apart the cv set which is different from the test set, as otherwise
the test set would become part of the training routine, and you might overfit your training data
without realising. By keeping it separate, you get a fair answer to the question - 'how well will
your data work on future data'.

Pretty much everybody does it this way. You can redo the split between train and cv - people often
use the term '10-fold cross-validation' where they do 10 different foldings then run the model 10
times to find the optimal k or smoothing parameter.

## Classification vs. Regression ##

So far, we've been talking about supervised learning via *classification*, where the target
labels/class is discrete (in our case binary).

Often we want to predict a continuous quantity, e.g. [; y_i ;] in [; [0, 1] ;]

Doing this is called 'regression'.

A regression problem could e.g. be to predict the weather/temperature which clearly involves a
continuous variable(s).

## Linear Regression ##

This isn't a binary thing anymore. More a relationship between two variables, e.g. one you know and
one you care about. You're trying to fit a curve which best represents the data.

You might try to fit a very non-linear curve, but again Occam's razor plays its part and you risk
overfitting should you get too carried away.

Data will be comprised of input vectors of length n which map to a continuous value, e.g.:-

    [; \begin{bmatrix}
        x_{11} & x_{12} & x_{13} & ... & x_{1n} & \to & y_1 \\
        x_{21} & x_{22} & x_{23} & ... & x_{2n} & \to & y_2 \\
        ...    & ...    & ...    & ... & ...    & ... & ... \\
        x_{m1} & x_{m2} & x_{m3} & ... & x_{mn} & \to & y_m \\
       \end{bmatrix} ;]

The difference between this + the classification case is that the y's are now continuous.

Once again we're looking for:-

    [; y = f(x) ;]

In linear regression we're looking at:-

    [; f(x) = w_1x + w_0 ;]

We could also do:-

    [; f(x) = wx + w_0 ;]

Where w and x are both vectors and we're calculating their inner product. However for now let's just
consider the 1-dimensional case.

## More Linear Regression ##

What are we trying to minimise?

We're trying to minimise the 'loss function'. The loss function gives us the amount of 'residual
error' after trying to fit the data as well as we possibly can.

The residual error is the sum of all training examples, minus our prediction, squared:-

    [; LOSS = \sum_j(y_j - w_1x_j - w_0)^2 ;]

This is the quadratic error between target labels and what our best hypothesis can produce.

We can write the minimisation problem as follows:-

    [; w_1^*=argmin_w L ;]

i.e. the arg min of the los over all possible vectors W.

## Quadratic Loss ##

The problem of minimising quadratic loss can be solved in closed form.

Let's look at the one-dimensional case:-

    [; L = min_w \sum(y_i - w_1x_i-w_0)^2 ;]

Minimum of this is obtained by setting the derivative of this to 0.

Partial derivation:-

    [; \frac{\partial L}{\partial w_0} = 0 = -2\sum(y_i - w_1x_1-w_0) ;]

Thus:-

    [; \sum y_i - w_1 \sum x_i = Mw_0 ;]

Hence:-

    [; w_0 = \frac{1}{M} \sum y_i - \frac{w_1}{M}\sum x_i ;]

Now if we look at [; w_1 ;]:-

    [; \frac{\partial L}{\partial w_1} = -2 \sum(y_i - w_1x_i - w_0)x_i = 0 ;]
    [; \sum_i x_i y_i - w_0 \sum x_i = w_1 \sum x_i^2 ;]
    [; \sum x_i y_i - \frac{1}{M} \sum y_i \sum x_i - \frac{w_1}{M}(\sum x_i)^2 = w_1 \sum x_i^2 ;]

Skipping some derivation:-

    [; w_1 = \frac{M\sum x_i y_i - \sum x_i \sum y_i}{M \sum x_i^2 - (\sum x_i)^2} ;]

## Problems with Linear Regression ##

Linear regression works well if the data is approximately linear.

Examples of bad situations to use linear regression:-

* Non-linear data - clearly a linear curve is not going to fit non-linear data.
* Outliers - Since we're minimising quadratic error, outliers have a hefty impact on our modelled
  curve. A very bad match is one where you

<img src="http://codegrunt.co.uk/images/ai/5-problems-with-linear-regression-1.png" />

### Logistic Regression ##

* As x tends to infinity, so does y. This is not always applicable. E.g. with weather, it's not
  correct to assume that the weather will simply become hotter or cooler. A means of dealing with
  this is to use 'logistic regression'.

There is a little more complexity here, represented by the following function:-

Given f(x), our linear function, then the output of logistic regression is obtained from:-

    [; z = \frac{1}{1 + e^{-f(x)}} ;]

This function returns a range of values from 0 - 1.

## Linear Regression and Complexity Control ##

Another problem is related to 'regularisation' or 'complexity control'. Sometimes we want a less complex model.

We achieve this via:-

    LOSS = LOSS(data) + LOSS(parameters)

    [; \sum_j(y_j - w_1x_j - w_0)^2 + \sum_i |w_i|^p ;]

Where the second function penalises the parameters.

Our regularisation term might be 'pulling' terms towards zero, along the circle if you use quadratic
error. This is done in a diamond shaped way for 'L1 regularisation'.

L1 regularisation has the advantage that it tends to make parameters 'sparse', e.g. you can drive
one of the parameters to zero. In the L2 case, parameters tend not to be as sparse, so often L1 is
preferred. E.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-linear-regression-and-complexity-control-1.png" />

## Minimising Complicated Loss Functions ##

How do we minimise more complicated cost functions? We generally have to resort to iterative methods
to minimise more complicated cost functions.

### Gradient Descent ###

We start with an initial guess, e.g. [; w_0 ;], then we update according to:-

    [; w^{i+1} \leftarrow w^i - \alpha \triangledown_w L(w^i) ;]

## Question 1 ##

Nothing to comment on.

## Question 2 ##

Nothing to comment on.

## Answer ##

Global minimum can be reached, but have to be careful to make sure the learning rate becomes smaller
and smaller over time, otherwise the prediction might bounce between values around the global
minimum and never reach it properly.

Can end up at local minimums.

Gradient descent is fairly universally applicable to more complicated problems, but you have to
check for local minima. Many optimisation books will describe tricks as to how to achieve this, we
won't go into it here.

## Gradient Descent Implementation ##

How do we minimise our cost function using gradient descent?

In our linear regression problem we use the following function (which we know we have a closed-form solution to):-

    [; L = \sum_j(y_j - w_1x_j - w_0)^2 ;]

    [; \frac{\partial L}{\partial w_1} = -2 \sum_j(y_j - w_1 x_j - w_0)x_j ;]
    [; \frac{\partial L}{\partial w_0} = -2 \sum_j(y_j - w_1 x_1 - w_0) ;]

Where we iteratively obtain our w values from:-

    [; w_1^m \leftarrow w_1^{m-1} - \alpha \frac{\partial L}{\partial w_1}(w_1^{m-1}) ;]
    [; w_0^m \leftarrow w_0^{m-1} - \alpha \frac{\partial L}{\partial w_0}(w_0^{m-1}) ;]

## Perceptron ##

There are different ways to apply linear functions to machine learning. Not only used for
regression, but also for classification. Particularly used by an algorithm called the 'perceptron'
algorithm.

Very early model of a neuron, invented in the 1940's.

Say we had positive and negative samples, then a 'linear separator' is a linear equation which
separates the two. Clearly not all datasets have such a separator, however some do. E.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-perceptron-1.png" />

Let's start with our linear equation:-

    [; w_1 x + w_0 ;]

If higher dimensional, then [; w_1 ;] and x might be vectors. Anyway, never mind! ;-)

We can obtain class labels as follows:-

    [; 1 $ if $ w_1x + w_0 \geq 0 ;]
    [; 0 $ if $ w_1x + w_0 < 0 ;]

Which gives us our linear separation classification equation:-

    [; f(x)=\left\{\begin{matrix}
    1 $ if $ w_1x + w_0 \geq 0 \\
    0 $ if $ w_1x + w_0 < 0
    \end{matrix}\right. ;]

Perceptron only converges if the data is linearly separable, and if so then in converges to a linear
separation of the data, which is quite amazing :-)

Perceptron is iterative + similar to gradient descent.

### Perceptron Update ###

Start with random guess for:-

    [; w_0, w_1 ;]

Then update via the following:-

    [; w_i^m \leftarrow w_i^{m-1} + \alpha (y_j - f(x_j)) ;]

This is an 'online' learning rule, i.e. we don't need to process in batch, process one bit of data
at a time. Might need to go through the data many times (the j in the second part of the equation is
not an error).

This method gives us the ability to adapt our weights in proportion to the error.

If we are dead-on, i.e. our function is entirely accurate, then we make no adjustment to w, however
if not, then we make a minor correction according to a small learning rate, [; \alpha ;].

Perceptron converges to a correct linear separator if one exists.

The case of linear separation has been receiving a lot of attention in machine learning recently.

You can make different choices of linear separator, not always totally clear-cut, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-perceptron-2.png" />

Intuitively, b is a good choice, because a and c are too close to the examples such that new
examples might end up crossing the line, whereas b does not stray too close.

The area between the closest training data and the linear separator.

The margin is very important - there are is an entire class of algorithms which try to maximise the
margins, e.g.:-

* Support vector machines / SVMs
* Boosting

Very frequently used in real classification machine learning tasks.

A support vector machine derives a linear separator and actually maximises the margin. It picks a
linear separator which acts to maximise this margin. Has a robustness advantage over a typical
perceptron.

You can solve for this by solving a quadratic problem.

One of the nice aspects of SVMs is that they use linear techniques to solve non-linear
problems. E.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-perceptron-3.png" />

We use a 'kernel trick' to solve this, as clearly a linear solution won't separate positive and
negative values as we would like.

We define a third term, [; x_3 ;]:-

    [; x_3 = \sqrt{(x_1)^2 + (x_2)^2} ;]

i.e. distance from origin. This trick can help you apply non-linear techniques to a linear system,
extremely useful and quite a deep insight to see that this can help. Numerous papers out in the
literature.

Mathematically done using a 'kernel' - out of scope of class, but very powerful and useful.

### Linear Methods ###

* Regression vs. Classification
* Exact Solutions vs. Iterative Solutions
* Smoothing
* Non-Linear Problems

## k Nearest Neighbours ##

The k-nearest-neighbours method is a non-parametric machine learning method.

So far we've been talking about parametric approaches.

Parametric methods have parameters, and the number of parameters is constant + independent of
training set size (for any fixed dictionary if applicable).

Non-parametric methods - number of parameters can grow (in fact, can grow a lot over time).

## kNN Definition ##

K-Nearest-Neighbours - very simple algorithm:-

* Learning Step: memorise all data.
* Label new example: If a new example comes along whose input value you know, which you wish to
  classify, find k nearest neighbours + return the *majority* class label as your final class label
  for the example.

E.g.:-

<img src="http://codegrunt.co.uk/images/ai/5-knn-definition-1.png" />

Here we label according to different k:-

* k=1 +
* k=3 +
* k=5 -
* k=7 -
* k=9 +

## k as Smoothing Parameter ##

As with LaPlacian smoothing before, k is a smoothing function which makes the data less scattered.

The 'Voronoi graph' shows the boundaries between positive and negative labelling by the kNN method.

k is a regulariser - larger k smoothes the output, but again there is a trade-off as things can get
misclassified. Trade-off between complexity and goodness of fit.

## Problems with kNN ##

Problems are:-

* Very large data sets

Lengthily searches for k nearest neighbours. Can be mitigated somewhat, via 'kdd trees', etc. -
makes search logarithmic.

* Very large feature spaces

Computing kNN, as the feature space of the input vectors increases becomes increasingly difficult +
the tree structure becomes more brittle.

The more features, the easier it is to select something far away, and soon it gets out of control
(e.g. exponential).

kNN works well for perhaps 3-4 dimensions, more is dodgy.

## Congratulations ##

Learnt a lot, focused on supervised machine learning - input vectors and given output labels, and
the goal is to predict the output labels given the input vectors.

Also looked at parametric models like naive Bayes, or non-parametric models like kNN. Looked at
classification vs. regression.

One of the more interesting parts of AI, dealing with big datasets. Only becoming more important.
