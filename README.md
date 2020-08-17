# Lane-Line-Analyzer

Lane line detection is crucial for developing algorithms for autonomous driving robots, self-driving cars or devices which are designed to keep drivers safe liek ADAS (Advanced Deiver Assistance Systems). The detection algorithm must be robust enough to face changing light and weather conditions, , lots os noisy like others vehicles on the road, different type of roads among others. Once you have a good lane detection one of the features can be implemented is the LDW (Lane Departure Warning). The driver might be warned up when the vehicle is moving away from the center of the lane.
However, to design a robust detector is not an easy task. You have to play around with a huge variety of conditions, for instance luminosity inside a tunnel although dark is not the same darkness of the road or even in a city. At the same time, a detector which is doing a very good job on the road with no cars might not does the same in a tottaly crowded road.

When you are writing your detector, you have to build your image pipeline, where in one side the original image comes in, frame by frame, and the another side a processed image comes out, ready to be analyzed by your detection mechanism.
Actually an image pipeline is sequential of image transformations, here is the place where you will use OpenCV extensively.  During those transformations OpenCv will require from you to define lots parameters and thresholds and much likely variations of those parameters might highlight or not the lanes depending of the luminosity conditions for example.

It means, even playing slighty changes on those parameters, they mith result to better output images in which can offer better perfomance to the lane line detector mechanism, so, as much as you learn about how the changes of the parameters might affect the quality of the detection more and more you can come up with heuristics which can handle the several different conditions.

It is exaclty what the Lane Line Analyzer project is about. Here you will find the image Pipeline and the detection mechanism from the NanoDegree Advaced Lane Line course plus three additional features which will allow you analyze and learn the effects of the changes in the final results.

## 1	- Canvas Object
Allow you create a kind of Canvas and add into it all images you want to analyze at the same time from your image pipeline. The canvas object will render the selected images allowing you analyze all images together and see the effects of the changes for each step of your process.
## 2 – Parameters Selector Screen
Simple screen with the most commom parameters in use by the image pipeline and lane line detector implemented in the project. You can use the screen to change the threshold and values and easily check what happened.
## 3 - DVR Control
It is a simple keyboard control which allow you pause (Space) the video or walk through it frame a frame using rewind (R) of Forward (F).

Feel free to share with me your finding about parameters and/or implementations of  specific conditions.

Enjoy!

