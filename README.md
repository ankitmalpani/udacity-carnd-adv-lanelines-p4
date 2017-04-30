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
[image18]: ./sample_images/undistort1.png "Undistort 1"
[image19]: ./sample_images/full_search.png "Full Search"
[image20]: ./sample_images/margin_search.png "Margin Search"
[video1]: ./lane_lines.mp4 "Video"

### Code structure & Notes
The main notebook submission(advanced_lane_detection_pipeline.ipynb) is organized a bit differently for some cleanliness and ease of understanding (especially after a long trial-error-test-repeat sort of a project). This will serve as a good index to the reader. The notebook is broadly divided into following segments:
1. Imports - This is where I import all the necessary libs
2. Camera calibration - This is where we load camera calibration images and use to calibrate the camera
3. Function definitions - This is the meat of the work. I have organized all methods together so it is easier to follow and much cleaner python code for future use
4. Pipeline - This where I have one method that consists of the main pipeline.
5. Testing cells - This is where I have re-run and tested some functions and displayed various images and charts
6. Video generation
##### Note:
You'll notice images titled 'First Image' & 'Second Image' in the doc. This is because of a generic method `display_two_images()` I wrote to display two images at once while working and testing on this project.

In the following segments I'll describe various steps I took along this project to reach the final output

### Camera Calibration
#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
Camera calibration was describe very well in the lecture along with basic exercises. I took this as my starting point for calibrating and undistorting. This consisted of some basic steps:
1. I first loaded all the chessboard calibration images
2. Examining the images - I realized that we had chessboards mostly of 9x6 dimensions. This helped me set the nx, ny values for our process.
3. As described in the lecture, I then defined the standard/initial objPoints which will be used to calibrate the camera.
4. I used OpenCV's `findChessboardCorners()` method to generate imagepoints for each calibration image provided. (I ignored the ones that won't find corners in the final submission as I didn't see a real output difference)
5. Finally, I used OpenCV's `calibrateCamera()` method to generate the camera matrix and distortion coefficients. (This used a list of obj_points and img_points)
6. I then use OpenCV's `undistort()` method to undistort images in one of the function definitions `image_of_interest()`. This can be seen in the cell with all function defintions as described above.
7. Note: Along the process I also used OpenCV's `drawChessboardCorners()` method to display and examine the accuracy of corners generated.

![alt text][image12]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one (Minor changes around the corners and edge between car and lane can be noticed):
![alt text][image18]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
After plenty of different combinations of Sobel XY thresholds , Sobel magnitude threshold, Sobel directional threshold, HLS color thresholds, the final combination as seen in method `get_thresholded_image()` includes:
1. Sobel thresholding in X direction limits 100, 255 (This can be seen in `abs_sobel_thresh()`)
2. Sobel directional thresholding using limits .5, 1.5 (This can be seen in `dir_threshold()`)
3. An RG threshold using limit 150 (This can be seen in `get_thresholded_image()`)
4. An L channel threshold for an HLS image using limits 120, 255 (This can be seen in `get_thresholded_image()`)
5. An S channel threshold for an HLS image using limits 100, 255 (This can be seen in `get_thresholded_image()`)

#### Note: All the used threshold numbers and other variables can be see in the first few lines of `pipeline()` function

![alt text][image1] ![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform can be seen in `perspective_transform()`. This method takes in an image, src/dst vertices and also a flag for inverse transform - which we use later in the project. After a few trial-and-error steps, I chose the hardcode the source and destination points in the following manner as seen in `pipeline()`:

```python
persp_trans_src = np.float32([[220,720],[1110, 720],[722, 470],[570, 470]])
persp_trans_dst = np.float32([[320,720],[920, 720],[920, 1],[320, 1]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image14] ![alt text][image15]

#### Note: At this time, I should note, the best way to follow how all the thresholding, masking, transforming works is the method `image_of_interest()`

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
This piece turned out to be tricker that I had initially expected. It involved multiple iterations to get to the right histograms for images, that were cleaner and not too choppy. I started out with implementing a full sliding window search only and then I also did some testing using sample code that convolves the images as well. It was much later in the project when I started implementing margin_search (when we have already seen previous images) and line averaging, that I had to keep coming back to this code. My code can be seen in following methods which are pretty similar to what is described in the lectures.
1. `full_sliding_window_search()`
2. `margin_search()`
3. `get_line_predictions()`
4. `get_histogram()`

![alt text][image8] ![alt text][image9]
![alt text][image19] ![alt text][image20]

In the method `pipeline()`, code exists to figure out frames where no lines could be found by the sliding window search, or if the line gap was an outlier - which prompted a throw-away frame(see Note after this section). In such scenarios we fall back to the previous frame for now (This probably could be improved).

#### Note on averaging and smoothing:
It is important to note that the final method `pipeline()` has the logic that maintains a running average gap between two lines and applies limits to discard anything that goes beyond that average.
It is also important to note the method `get_averaged_line()` which basically computes and returns line predictions based on the average of last 10 frames. This probably could be improved( or at least fine tuned for an optimal average calculation, maybe a smaller number of frames)


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This code can be seen in method `curvature_radius()` . This is pretty much implementation of formula from the lectures and the pixel-meter conversion has noted in the lectures.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I basically implemented this using the above mentioned method `perspective_transform()` with the flag `inv=True`. This is done in the pipeline itself (around the 4th last line). Here are some sample output (single image output of the whole pipeline)

![alt text][image11] ![alt text][image3]

---

### Pipeline (video)

Here's a [link to output video](./lane_lines.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
This was quite an interesting project. There were may trial-error-test-repeat cycles through out this project. I believe, what I have can be improved in various ways:
1. This pipeline won't correctly work on the challenge videos. I would love to start exploring the challenge videos and fine tune this pipeline further
2. I would also like to explore some tricks we learned in the first project and try to see if we can benefit from them. I would like to solve some processing issues with canny_edge and see how they compare.
3. I think smoothing can be improved a lot on my project. It will be a good exercise to go back and optimize some numbers that were chosen based off of some test images.
4. I would love to read more on the topic of lane detection in general. It would be interesting to know what is used in practice and how many of the numbers needed for this project are calculated and used for accurate detection

#### Some additional resources I referred to during the project:
* https://chatbotslife.com/robust-lane-finding-using-advanced-computer-vision-techniques-46875bb3c8aa
* https://discussions.udacity.com/t/pixel-space-to-meter-space-conversion/241646
* https://discussions.udacity.com/t/any-information-to-check-radius-of-curvature-and-off-of-center/241681/2
* https://discussions.udacity.com/t/trying-to-understand-sanity-check-validation-criteria-and-line-smoothing/239174
