AI Notes
========

Note this page uses [tex the world](http://thewe.net/tex/) notation extensively.

Unit 16 - Computer Vision I
---------------------------

## Introduction ##

We've talked about basic methods of AI, now we'd like to look at applications, specifically for this
unit - computer vision.

Computer vision is a wide field related to making sense out of camera images or video.

Many devices are equipped with cameras (not just specifically cameras themselves), so it's become a
really important subfield of AI.

Going over some of the very basics, to, for example, classify images using feature extraction and
other techniques and also to look at some of the 3D tasks such as 3D constructions.

What is a camera? Come in all sizes and shapes. E.g. Nikon D3 camera - heavy, but produces great
pictures. Alternative is a mobile phone. In all the cameras there are a lens and a chip - a lens
focuses the light on the chip - this raises the question - how do a lens and a chip really work?

## Image Formation ##

The science of how images are created using cameras is called 'image formation' where formation
simply means the way an image is being captured.

Perhaps the easiest model of camera is a 'pinhole camera':-

<img src="http://codegrunt.co.uk/images/ai/16-image-formation-1.png" />

Here we have a very small hole through which light travels, projecting for example, a person, with
the slight projecting as such to invert the person on the projection plane of the camera chip.

Here we have some basic maths relating to the projection of the image:-

* X is the physical height of the object
* x the height of the projection (here -x because it's projected in the opposite direction of the
  real person)
* Z is the distance of the object to the camera plane.
* f is the focal distance/length of the pinhole to the projection plane.

We use the fact that the triangles are equal and thus have the same gradient of hypotenuse to
obtain:-

    [; \frac{X}{z} = \frac{x}{f} ;]

This leads to:-

    [; x = X \frac{f}{z} ;]

Which is interesting - the further away an object is, the smaller it appears. The larger the focal
length, the larger the object appears. Of course the size of the object itself has an influence on
how large it appears.

## Project Length Question ##

Nothing to note.

## Focal Length Question ##

Nothing to note.

## Range Question ##

Nothing to note.

## Perspective Projection ##

We've learnt about perspective projection - the projected size of an object scales with distance.

If you take an object and move it further away, it'll appear smaller.

<img src="http://codegrunt.co.uk/images/ai/16-perspective-projection-1.png" />

This is an intuitive result, clearly.

## Vanishing Points + Question ##

Actual camera images have two dimensions. The perspective law applies to them both, so considering x
and y:-

    [; x = X \frac{f}{z} ;]
    [; y = Y \frac{f}{z} ;]

E.g.

<img src="http://codegrunt.co.uk/images/ai/16-vanishing-points-question-1.png" />

One of the interesting consequences of perspective projection is that parallel lines in the world
seem to result in vanishing points.

## Lenses ##

A limitation of a pinhole camera is that only very few rays of light hit the plane of the imager
through the pinhole, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/16-lenses-1.png" />

This means a pinhole approach is only applicable to very bright scenes. Also, as you make the gap
smaller and smaller in order to increase the focus on the image plane, you eventually result into
light diffraction, which puts a limit on how small the hole can be.

However, if you put a lens on the camera, then all rays will be projected to one point, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/16-lenses-2.png" />

This all depends on the object being in focus, then the resulting projections don't match up, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/16-lenses-3.png" />

There is an equation which governs all of this:-

    [; \frac{1}{f} = \frac{1}{Z} + \frac{1}{z} ;]

E.g.:-

<img src="http://codegrunt.co.uk/images/ai/16-lenses-4.png" />

## Computer Vision ##

We've learnt the law of perspective projection (which is really important) and the law which
determines when things are in focus (which we don't really care about that much).

Computer vision is concerned with:-

* Classifying objects
* 3D Reconstruction
* Motion analysis

## Invariance Question A-E ##

Object recognition is the task of having an object in an image and wanting to understand what the
nature of the object is - e.g. identifying a plane, car amongst other objects.

A key concept in object recognition is invariance. There are natural variations in the object that
don't affect the nature of the object - you want to be invariant to the natural variations.

Different types of variations:-

* Scale
* Illumination
* Rotation
* Deformation
* Occlusion
* View Point

## Final Invariance Type ##

View point invariance - different view points of an object change how they appear. View
point/vantage point invariance is one of the hardest to deal with as the appearance of the object
might vary a great deal.

## Importance of Invariance ##

We are talking about these invariances because they are very important to computer vision.

## Greyscale Images ##

In computer vision we rarely use colour images, tend to use black and white images. This reduces
information as compared to colour images, however it turns out greyscale images are more robust to
lighting variations.

We can represent greyscale images are a matrix of values, each value representing the grey level at
that point.

Each greyscale value is in the range of 0 to 255 i.e. a byte. 0 is black, 255 is white, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/16-greyscale-images-1.png" />

A colour image would be similar but consisting of 3 different values per pixel, corresponding to
red, blue and green or some other encoding.

## Extracting Features + Question ##

One of the most basic things you can do is to extract features. For example, considering how to
differentiate light and dark parts of an image. One approach to this is to essentially put an
mask of two cells over the whole image, then add the left-hand value, and subtract the right-hand
value, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/16-extracting-features-question-1.png" />

Doing this means that when there is little difference, the value returned is low, however when there
is a large difference, the value is high. Applying over matrix:-

<img src="http://codegrunt.co.uk/images/ai/16-extracting-features-question-2.png" />

The values in the mask indicate a high probability of there being a vertical edge feature between
the columns the mask refers to.

Our 'kernel' will only find vertical edges, and not horizontal edges.

## Linear Filter ##

We've here applied a 'linear filter' - taken an image, I, and applied a kernel, g, to it, to obtain
a new image, I', e.g.:-

<img src="http://codegrunt.co.uk/images/ai/16-linear-filter-1.png" />

The resultant image I' is smaller, however we could avoid that if necessary by assuming all the
values 'surrounding' the image are 0.

The equation we are applying is:-

    [; I'(x, y) = \sum_{u, v} I(x-u, y-v) g(u, v) ;]

This is essentially describing what we've done before - we go over both fields u and v, shift the
image by them, and multiply the image value by the kernel.

The 'convolution' operation is often known as a 'linear' operation.

## Horizontal Edge Question ##

Nothing to note.

## Vertical Filter Question ##

Nothing to note.

## Filter Results ##

Original image:-

<img src="http://codegrunt.co.uk/images/ai/16-filter-results-1.png" />

Vertical filter:-

<img src="http://codegrunt.co.uk/images/ai/16-filter-results-2.png" />

Vertical filter:-

<img src="http://codegrunt.co.uk/images/ai/16-filter-results-3.png" />

## Gradient Images ##

Horizontal and vertical convolution can be represented as follows:-

<img src="http://codegrunt.co.uk/images/ai/16-gradient-images-1.png" />

If we want to find all edges, then we can combine both of these:-

    [; E = \sqrt{(I_x)^2 + (I_y)^2} ;]

Which gives us this result:-

<img src="http://codegrunt.co.uk/images/ai/16-gradient-images-2.png" />

## Canny Edge Detector ##

A Canny edge detector gives better edge detection:-

<img src="http://codegrunt.co.uk/images/ai/16-canny-edge-detector-1.png" />

Which gives far sharper edge detection. Not only does it find gradient magnitude, it also traces
areas and finds local optima and tries to connect them such that there is always a single edge. When
multiple edges meet, the canny edge detector has a hole, however when edges are single edges then
the Canny edge detector traces them very nicely. Named after John canny at UC Berkeley.

## Other Masks ##

There are different kernels one can use:-

<img src="http://codegrunt.co.uk/images/ai/16-other-masks-1.png" />

## Prewitt Mask Question ##

Nothing to note.

## Gaussian Kernel Question ##

A Guassian kernel is one which has maximum value at the centre of the matrix, but whose values fall
exponentially towards the sides. This blurs an image as you are essentially combining neighbourhoods
of pixels into one. The larger the neighbourhood, the more blurred the image.

## Reasons for Gaussian Kernels ##

Why would we do this:-

* Down-sampling - If we want to reduce the size of an image, it's better to blur with a Gaussian
  rather than picking each nth pixel. This is because of the issue of aliasing - we might pick by
  chance pixels which correspond to something somewhat irregular, e.g. a checkerboard, picking only
  the black elements.
* Noise reduction - Noise might make it hard to find image gradients - by blurring the image first
  you get a smoother result with far less noise.

We can apply a Gaussian kernel and a gradient kernel thus:-

    [; I \otimes f \otimes g ;]

Convolution is associative, so this can be represented as:-

    [; I \otimes (f \otimes g) ;]

Where we obtain a Gaussian gradient kernel here. Can have a single linear kernel that does smoothing
and find gradients at the same time.

## Harris Corner Detector ##

Sometimes we want to find corners rather than edges. Corners have an advantage over edges in that
edges aren't localisable - they could be anywhere on an edge, but corners can be localised which is
useful in computer vision.

Looking at a Harris corner detector we get results such as:-

<img src="http://codegrunt.co.uk/images/ai/16-harris-corner-detector-1.png" />

The algorithm is quite simple. In a corner there are lots of vertical and horizontal edges. If we
sum the convoluted result of a horizontal edge detection kernel and a vertical edge detection kernel
and both values are large, we likely have an edge, e.g.:

<img src="http://codegrunt.co.uk/images/ai/16-harris-corner-detector-2.png" />

This is also generalised to handling rotation cases by de-rotating the image using Eigenvalue
decomposition, e.g.:-

<img src="http://codegrunt.co.uk/images/ai/16-harris-corner-detector-3.png" />

We end up with 2 eigenvalues. If both eigenvalues are large, then we have a corner. If we apply this
to every pixel in the image and take the local maxima of the result where the results are both large
then we have a Harris corner detector.

## Modern Feature Detectors ##

Modern feature detectors extend Harris corner into much more advanced features. They are:-

* Localisable
* Possessing of unique signatures which identify items which are typically invariant to lighting,
  orientation, translation and size variance.

Typical methods used are:-

* HOG = Histogram of Oriented Gradients
* SIFT = Scale Invariant Feature Transform

Each of these perform corner detection then reduce the variances by extracting statistics which are
invariant to rotation and scale and certain perspective transformations.

Applying SIFT to the bridge photo:-

<img src="http://codegrunt.co.uk/images/ai/16-modern-feature-detectors-1.png" />

## Conclusion ##

We've gone over the basics of computer vision. We've talked about images + how they are formed,
perspective projection as a mathematical tool for understanding how cameras perceive images. We've
talked a lot about features and invariances, the type of thing that might affect the appearance of a
feature in the camera image, looked at means of extracting edges, corners and more sophisticated
features such as SIFT features. Very basic stuff - almost everyone who does computer vision
preprocesses images by feature extraction. 
