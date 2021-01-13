# ARSpectator-Localisation


* Desktop - C++/OpenCV implementation, loads a single image frame from disk and performs localisation once the user has selected a pitch line.
* Mobile - C# interface and C++ native plugin, to be used inside a Unity/Vuforia project.
* Scraps - Old versions of the desktop application. Only used for testing improvements made in across iterations.


# Desktop Application

The code for our desktop application is split into four different files:
* Main.cpp - Handles image loading, user input, execution of the localisation pipeline and the display of debug output images.
* LineDetection.cpp - Handles preprocessing of the image (pitch extraction) and the initial line detection step.
* LineClustering.cpp - Functions for the clustering of detected lines, fitting of one line per cluster, and matching of lines between the model and the image.
* Support.cpp - Various small support functions.

Things to change when modifying for different environments:

main.cpp - templateLines. This is a set of vectors that represent the 2D pitch model. 
These lines are stored in a specific order: All vertical pitch lines in left to right order, followed by the top vertical line, and then the bottom vertical line.

Three different line sets are used depending on whether the use selected the leftmost, center, or rightmost lines (using left, middle, and right mouse clicks respectively). 

lineDetection.cpp - hue threshold for pitch segmentation. This is set manually inside the GetLines() function (line 61) if the autoThresh parameter is set to false.
