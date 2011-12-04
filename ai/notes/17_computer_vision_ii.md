AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 17 - Computer Vision II
----------------------------

## Introduction ##

This unit is all about 3d vision:-

<img src="http://codegrunt.co.uk/images/ai/17-introduction-1.png" />

We lose something in the projection - range (sometimes depth/distance - the task is to recover this from single or multiple camera images.

## Depth Question ##

Given a simple image, and given parameters of the camera like focal length + any other parameters,
can we recover the depth of a scene?

Sometimes we can - where we know the size of the object beforehand we can, otherwise we can't.

## Stereo Question ##

Given 2 identical cameras for which we know the baseline can we now recover the depth of a scene?

Again, sometimes - when the two cameras observe something vertical, then they will be slightly
displaced from one another, however when two cameras observe something horizontal, then we can't
necessarily tell which parts of the object correspond to one another in the two images - this is due
to the 'aperture effect'.

So in summary - there are certain degenerate cases where stereo won't work.

## Solve Depth + Question ##

Consider the following stereo rig:-

<img src="http://codegrunt.co.uk/images/ai/17-solve-depth-question-1.png" />

Here B is the baseline, f is the focal length, x1 and x2 are measured distances on the resultant
image, and Z is the depth.

We are basically applying simliar triangles again to obtain:-

    [; Z = \frac{fB}{x_2-x_1} ;]

## Change In X Question ##

We can say:-

    [; x_2 - x_1 = \Delta x ;]

So we end up with:-

    [; Z = \frac{fB}{\Delta x} ;]

Clearly we can rearrange this formula to obtain:-

    [; \Delta x = \frac{fB}{Z} ;]

## Focal Length Question ##

Can also rearrange to obtain f:-

    [; f = \frac{Z \Delta x}{B} ;]

## Correspondence Question ##

In order to find a corresponding point in one camera knowing the position in another we need only
consider the position along a one-dimensional line, as not knowing the depth, a point can only
project along a specific line:-

<img src="http://codegrunt.co.uk/images/ai/17-correspondence-question-1.png" />

## Determine Correspondence Question ##

The general correspondence problem is given when there are two identically looking points in the
scene with different depths, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/17-determine-correspondence-1.png" />

We have to be careful to ensure we correspond points in the image with the correct real-world
points, otherwise we can end up with 'phantom points', e.g. p1' and p2' above.

An example of the correspondence problem:-

<img src="http://codegrunt.co.uk/images/ai/17-determine-correspondence-2.png" />

We can determine correspondence by matching small image patches and matching features.

## SSD Minimisation ##

Here are the original stereo images shown again:-

<img src="http://codegrunt.co.uk/images/ai/17-ssd-minimisation-1.png" />

We are comparing the patch shown on the left, and comparing to potential matches on the right,
mapping the *sum of square difference*, or SSD for each, which is minimised when the two patches are
alike. The SSD is also known as disparity here.

To determine SSD, we follow the process of normalising both candidate patches so the average
brightness is zero, we take the difference of each pixel, then square them, and then sum all of
these pixels - these result in our SSD value:-

<img src="http://codegrunt.co.uk/images/ai/17-ssd-minimisation-2.png" />

The smaller the SSD value, the closer the two patches correspond.

This is a very common technique for comparing what are called image templates, where the left image
is a template, and you are searching the right image for the template. Often you obtain results
similar to:-

<img src="http://codegrunt.co.uk/images/ai/17-ssd-minimisation-3.png" />

Where the minimum value is where the template matches.

## Disparity Maps ##

Looking at the original building image again, and finding the best possible match for every patch,
we obtain a 'disparity map':-

<img src="http://codegrunt.co.uk/images/ai/17-disparity-maps-1.png" />

In the foreground the disparity is much larger, but in the background or where there are few
features like the path the disparity reduces.

## Context Question ##

Want to discuss correspondence - we've determined that searching for correspondence means we search
along a single scanline, want to determine whether it's optimal to correspond individual patches
against one another independently of one another, or whether it makes sense to look at the context
of an entire scanline.

Consider:-

<img src="http://codegrunt.co.uk/images/ai/17-context-question-1.png" />

Due to occlusion, the right camera doesn't see the blue line.

## Alignment 1 + 2 Questions ##

We can assign a cost to mismatches, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/17-alignment-1-2-questions-1.png" />

Here the cost is 20, due to 2 occlusions.

## Dynamic Programming ##

The trickiest part of the approach taken above is to find the minimum cost. This is usually achieved
via dynamic programming.

In principle, there are exponentially many ways in which you can align pixels in the two images, but
in practice, you can get away with an [; n^2 ;] algorithm.

We represent scan line alignments as follows:-

<img src="http://codegrunt.co.uk/images/ai/17-dynamic-programming-1.png" />

Here we map correspondence by diagonal lines, and occlusions by horizontal/vertical lines along the
grid.

In order to find the best possible path, we determine the value of any point in the grid (i.e. point where lines meet) from the best possible means of getting there. So:-

    [; V(i, j) = min { match (i, j) + V(i-1, j-1) + occl + v(i-1, j) + occl + v(i, j-1) } ;]

We find the cheapest price for the full traverse of the path, and then trace back from that to find
the optimal path.

## Pixel Correspondence Question 1 + 2 ##

Nothing to note.

## Finding the Best Alignment ##

The best path is clearly determined by the cost of the various penalties we've assigned.

## Correspondence Issues ##

There are a few things which don't work very well, e.g. a foreground separate object will cause an
object immediately behind it to reverse its relative position in the two imagers - it might appear
to the left in the left imager and to the right in the right imager, which violates the order
constraint of the dynamic programming problem.

<img src="http://codegrunt.co.uk/images/ai/17-correspondence-issues-1.png" />

In the bottom example we see that each imager has a different occlusion boundary, so points might
seem to correspond when in fact they are different points on the object.

Note: Couldn't eliminate Dr. Thrun's hand here :-)

Another instance when things might go wrong is where the object has specular reflections, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/17-correspondence-issues-2.png" />

## Improving Stereo Vision ##

Here's an example of an attempt to improve stereo vision:-

<img src="http://codegrunt.co.uk/images/ai/17-improving-stereo-vision-1.png" />

Here a striped pattern is projected onto the object, where stripes are unequal - some stripes are
larger than others which makes it easier to determine correspondence. The previously discussed
dynamic programming algorithm can then be applied and it will provide improved results.

Similarly, we can project 'structured' light which improves the texture of poorly textured
objects. Another solution, used by Microsoft Kinect, is to project a laser and by triangulation
determine depth of objects.

There are also scanning laser range finders which send beams of light and measure the time taken for
the beam to return. Extensively used in both Google automated car work and many other places.

Laser range finding is an alternative to stereo vision.
