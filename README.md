# udacity-carnd-adv-lanelines-p4
Advanced lane finding using various computer vision techniques (Udacity CarND Project 4)

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./sample_images/threshold1.png "Threshold 1"
[image2]: ./sample_images/threshold2.png "Threshold 2"
[image3]: ./sample_images/straight1.png "Straight Line 1"
[image4]: ./sample_images/original1.png "Test 1"
[image5]: ./sample_images/original2.png "Test 2"
[image6]: ./sample_images/original3.png "Test 3"
[image7]: ./sample_images/original4.png "Test 4"
[image8]: ./sample_images/hist3.png "Histogram 3"
[image9]: ./sample_images/hist4.png "Histogram 4"
[image10]: ./sample_images/final1.png "Final 1"
[image11]: ./sample_images/final2.png "Final 2"
[image12]: ./sample_images/chess1.png "Chess 1"
[image13]: ./sample_images/chess2.png "Chess 2"
[image14]: ./sample_images/birdseye3.png "Birdseye 3"
[image15]: ./sample_images/birdseye4.png "Birdseye 4"
[image16]: ./sample_images/straight2.png "Straight 2"
[image17]: ./sample_images/straight_thresh2.png "Straight Threshold 2"

### Code structure
The main notebook submission(advanced_lane_detection_pipeline.ipynb) is organized a bit differently for some cleanliness and ease of understanding (especially after a long trial-error-test-repeat sort of a project). This will serve as a good index to the reader. The notebook is broadly divided into following segments:
1. Imports - This is where I import all the necessary libs
2. Camera calibration - This is where we load camera calibration images and use to calibrate the camera
3. Function definitions - This is the meat of the work. I have organized all methods together so it is easier to follow and much cleaner python code for future use
4. Pipeline - This where I have one method that consists of the main pipeline.
5. Testing cells - This is where I have re-run and tested some functions and displayed various images and charts
6. Video generation

In the following segments I'll describe various steps I took along this project to reach the final output

### Camera Calibration
#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
Camera calibration was describe very well in the lecture along with basic exercises. I took this as my starting point for calibrating and undistorting. This consisted of some basic steps:
1. I first loaded all the chessboard calibration images
2. Examining the images - I realized that we had chessboards mostly of 9x6 dimensions. This helped me set the nx, ny values for our process.
3. As described in the lecture, I then defined the standard/initial objPoints which will be used to calibrate the camera.
4. I used OpenCV's `findChessboardCorners` method to generate imagepoints for each calibration image provided. (I ignored the ones that won't find corners in the final submission as I didn't see a real output difference)
5. Finally, I used OpenCV's `calibrateCamera` method to generate the camera matrix and distortion coefficients. (This used a list of obj_points and img_points)
6. I then use OpenCV's `undistort` method to undistort images in one of the function definitions `image_of_interest`. This can be seen in the cell with all function defintions as described above.
6. Note: Along the process I also used OpenCV's `drawChessboardCorners` method to display and examine the accuracy of corners generated.

![alt text][image12] ![alt text][image13]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
