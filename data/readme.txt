This data set contains 30 adult C. elegans swimming in M9 buffer solution. Parameters of the data are:

Recording speed: 125 frames per second
Image size: 1024 x 1024 pixels
Image resolution: 3.97 um/pixel


Each skeleton_XX.mat file contains a portion of our analysis. Using image processing, we divide the worm body into 100 segments.  You probably are most interested in the following data:

XC - x-position of the centroid in each frame (1 x numFrames)
YC - y-position of the centroid in each frame (1 x numFrames)
kappa - curvature of each body segment in each frame (100 x numFrames)

Other things you may find useful:

XX - x-position of each body segment in each frame (100 x numFrames)
YY - y-position of each body segment in each frame (100 x numFrames)


If you have any questions, please let me know!

David