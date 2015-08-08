AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 18 - Computer Vision III
-----------------------------

## Structure from Motion Question ##

The idea of 'structure from motion' is to take a handheld camera, and move it around a 3d structure
and be able to recover or estimate the 3d coordinates of all the features in the world based on many
2d images, e.g.:-

<img src="https://ljs.io/img/ai/18-structure-from-motion-question-1.png" />

## Projection Question ##

Nothing to note.

## Structure from Motion Models ##

Nothing to note.

## SFM Maths ##

The mathematics of structure from movement are involved. Don't want to go into too much detail.

Here is our original projection model:-

<img src="https://ljs.io/img/ai/18-sfm-maths-1.png" />

Once we take into account that we are viewing the object from an arbitrary angle, we end up with:-

<img src="https://ljs.io/img/ai/18-sfm-maths-2.png" />

We have to sum over i and j and work to find the minimum. This is very much non-trivial.

Non-linear methods have been used extensively to attempt to solve this non-linear least squares
problem (SFM) problem, e.g.:-

* Gradient Descent
* Conjugate Gradient
* Gauss-Newton
* Levenberg Marquardt
* Singular Value decomposition (affine, orthographic)

## Recovered Unknowns Question ##

With:-

* m camera poses (motion)
* n points (structure)

We have :-

* 2mn constraints due to having x and y coordinates for each point from each pose,
* 6m + 3n unknowns - each camera pose has a 6d unknown about the rotation and translation of the
  camera, and each point has 3 unknown dimensions.

We want:-

    [; 2mn \leq 6m + 3n ;]

To be able to solve this.

How many unknowns cannot be recovered?

There are 7 - the absolute location and orientation of the coordinate system which consist of 6 (3
times 2) of the parameters, but also cannot recover *scale* since perspective means a scaled up
scene looks the same as one which is not.

We care about this because it determines whether the problem is 'well-posed' such that we can solve
it.

## Conclusion ##

Nothing to note.
