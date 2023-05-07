Download Link: https://assignmentchef.com/product/solved-machinelearning-exercise-4-neural-networks-learning
<br>
In this exercise, you will implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition. Before starting on the programming exercise, we strongly recommend watching the video lectures and completing the review questions for the associated topics.

To get started with the exercise, you will need to download the starter code and unzip its contents to the directory where you wish to complete the exercise. If needed, use the cd command in Octave/MATLAB to change to this directory before starting this exercise.

You can also find instructions for installing Octave/MATLAB in the “Environment Setup Instructions” of the course website.

<h2>Files included in this exercise</h2>

ex4.m – Octave/MATLAB script that steps you through the exercise ex4data1.mat – Training set of hand-written digits ex4weights.mat – Neural network parameters for exercise 4 submit.m – Submission script that sends your solutions to our servers displayData.m – Function to help visualize the dataset fmincg.m – Function minimization routine (similar to fminunc) sigmoid.m – Sigmoid function computeNumericalGradient.m – Numerically compute gradients checkNNGradients.m – Function to help check your gradients debugInitializeWeights.m – Function for initializing weights predict.m – Neural network prediction function

[<em>?</em>] sigmoidGradient.m – Compute the gradient of the sigmoid function

[<em>?</em>] randInitializeWeights.m – Randomly initialize weights

[<em>?</em>] nnCostFunction.m – Neural network cost function

<em>? </em>indicates files you will need to complete

Throughout the exercise, you will be using the script ex4.m. These scripts set up the dataset for the problems and make calls to functions that you will write. You do not need to modify the script. You are only required to modify functions in other files, by following the instructions in this assignment.

<h2>Where to get help</h2>

The exercises in this course use Octave<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> or MATLAB, a high-level programming language well-suited for numerical computations. If you do not have Octave or MATLAB installed, please refer to the installation instructions in the “Environment Setup Instructions” of the course website.

At the Octave/MATLAB command line, typing help followed by a function name displays documentation for a built-in function. For example, help plot will bring up help information for plotting. Further documentation for Octave functions can be found at the <a href="https://www.gnu.org/software/octave/doc/interpreter/">Octave documentation pages</a><a href="https://www.gnu.org/software/octave/doc/interpreter/">.</a> MATLAB documentation can be found at the <a href="https://www.mathworks.com/help/matlab/?refresh=true">MATLAB documentation pages</a><a href="https://www.mathworks.com/help/matlab/?refresh=true">.</a>

We also strongly encourage using the online <strong>Discussions </strong>to discuss exercises with other students. However, do not look at any source code written by others or share your source code with others.

<h1>1          Neural Networks</h1>

In the previous exercise, you implemented feedforward propagation for neural networks and used it to predict handwritten digits with the weights we provided. In this exercise, you will implement the backpropagation algorithm to <em>learn </em>the parameters for the neural network.

The provided script, ex4.m, will help you step through this exercise.

<h2>1.1        Visualizing the data</h2>

In the first part of ex4.m, the code will load the data and display it on a 2-dimensional plot (Figure 1) by calling the function displayData.

Figure 1: Examples from the dataset

This is the same dataset that you used in the previous exercise. There are 5000 training examples in ex3data1.mat, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image.

 — (<em>x</em>(1))<em>T </em>— 

— (<em>x</em>(2))<em>T </em>—

             ..



— (

The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. To make things more compatible with Octave/MATLAB indexing, where there is no zero index, we have mapped the digit zero to the value ten. Therefore, a “0” digit is labeled as “10”, while the digits “1” to “9” are labeled as “1” to “9” in their natural order.

<h2>1.2        Model representation</h2>

Our neural network is shown in Figure 2. It has 3 layers – an input layer, a hidden layer and an output layer. Recall that our inputs are pixel values of digit images. Since the images are of size 20 × 20, this gives us 400 input layer units (not counting the extra bias unit which always outputs +1). The training data will be loaded into the variables X and y by the ex4.m script.

You have been provided with a set of network parameters (Θ<sup>(1)</sup><em>,</em>Θ<sup>(2)</sup>) already trained by us. These are stored in ex4weights.mat and will be loaded by ex4.m into Theta1 and Theta2. The parameters have dimensions that are sized for a neural network with 25 units in the second layer and 10 output units (corresponding to the 10 digit classes).

<table width="527">

 <tbody>

  <tr>

   <td width="527">% Load saved matrices from file load(‘ex4weights.mat’);% The matrices Theta1 and Theta2 will now be in your workspace% Theta1 has size 25 x 401% Theta2 has size 10 x 26</td>

  </tr>

 </tbody>

</table>

Figure 2: Neural network model.

<h2>1.3        Feedforward and cost function</h2>

Now you will implement the cost function and gradient for the neural network. First, complete the code in nnCostFunction.m to return the cost.

Recall that the cost function for the neural network (without regularization) is

<em> ,</em>

where <em>h<sub>θ</sub></em>(<em>x</em><sup>(<em>i</em>)</sup>) is computed as shown in the Figure 2 and <em>K </em>= 10 is the total number of possible labels. Note that is the activation (output value) of the <em>k</em>-th output unit. Also, recall that whereas the original labels (in the variable y) were 1, 2, …, 10, for the purpose of training a neural network, we need to recode the labels as vectors containing only values 0 or 1, so that

<table width="290">

 <tbody>

  <tr>

   <td width="99">             1 0       <em>y </em>=  0 <em>,</em> … <sub></sub>0</td>

   <td width="70">      0 1        0 <em>,</em> … <sub></sub>0</td>

   <td width="37"><em>…</em></td>

   <td width="32">or</td>

   <td width="52">        0 0               0 <em>.</em> … <sub></sub>1</td>

  </tr>

 </tbody>

</table>

For example, if <em>x</em><sup>(<em>i</em>) </sup>is an image of the digit 5, then the corresponding <em>y</em><sup>(<em>i</em>) </sup>(that you should use with the cost function) should be a 10-dimensional vector with <em>y</em><sub>5 </sub>= 1, and the other elements equal to 0.

You should implement the feedforward computation that computes <em>h<sub>θ</sub></em>(<em>x</em><sup>(<em>i</em>)</sup>) for every example <em>i </em>and sum the cost over all examples. <strong>Your code should also work for a dataset of any size, with any number of labels </strong>(you can assume that there are always at least <em>K </em>≥ 3 labels).

<strong>Implementation Note: </strong>The matrix X contains the examples in rows (i.e., X(i,:)’ is the i-th training example <em>x</em><sup>(<em>i</em>)</sup>, expressed as a <em>n </em>× 1 vector.) When you complete the code in nnCostFunction.m, you will need to add the column of 1’s to the X matrix. The parameters for each unit in the neural network is represented in Theta1 and Theta2 as one row. Specifically, the first row of Theta1 corresponds to the first hidden unit in the second layer. You can use a for-loop over the examples to compute the cost.

Once you are done, ex4.m will call your nnCostFunction using the loaded set of parameters for Theta1 and Theta2. You should see that the cost is about 0.287629.

<em>You should now submit your solutions.</em>

<h2>1.4        Regularized cost function</h2>

The cost function for neural networks with regularization is given by

You can assume that the neural network will only have 3 layers – an input layer, a hidden layer and an output layer. However, your code should work for any number of input units, hidden units and outputs units. While we have explicitly listed the indices above for Θ<sup>(1) </sup>and Θ<sup>(2) </sup>for clarity, do note that <strong>your code should in general work with </strong>Θ<sup>(1) </sup><strong>and </strong>Θ<sup>(2) </sup><strong>of any size</strong>.

Note that you should not be regularizing the terms that correspond to the bias. For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix. You should now add regularization to your cost function. Notice that you can first compute the unregularized cost function <em>J </em>using your existing nnCostFunction.m and then later add the cost for the regularization terms.

Once you are done, ex4.m will call your nnCostFunction using the loaded set of parameters for Theta1 and Theta2, and <em>λ </em>= 1. You should see that the cost is about 0.383770.

<em>You should now submit your solutions.</em>

<h1>2          Backpropagation</h1>

In this part of the exercise, you will implement the backpropagation algorithm to compute the gradient for the neural network cost function. You will need to complete the nnCostFunction.m so that it returns an appropriate value for grad. Once you have computed the gradient, you will be able to train the neural network by minimizing the cost function <em>J</em>(Θ) using an advanced optimizer such as fmincg.

You will first implement the backpropagation algorithm to compute the gradients for the parameters for the (unregularized) neural network. After you have verified that your gradient computation for the unregularized case is correct, you will implement the gradient for the regularized neural network.

<h2>2.1        Sigmoid gradient</h2>

To help you get started with this part of the exercise, you will first implement the sigmoid gradient function. The gradient for the sigmoid function can be computed as

where

sigmoid(<em>.</em>

When you are done, try testing a few values by calling sigmoidGradient(z) at the Octave/MATLAB command line. For large values (both positive and negative) of z, the gradient should be close to 0. When z = 0, the gradient should be exactly 0.25. Your code should also work with vectors and matrices. For a matrix, your function should perform the sigmoid gradient function on every element.

<em>You should now submit your solutions.</em>

<h2>2.2        Random initialization</h2>

When training neural networks, it is important to randomly initialize the parameters for symmetry breaking. One effective strategy for random initialization is to randomly select values for Θ<sup>(<em>l</em>) </sup>uniformly in the range [

You should use This range of values ensures that the parameters are kept small and makes the learning more efficient.

Your job is to complete randInitializeWeights.m to initialize the weights for Θ; modify the file and fill in the following code:

<table width="527">

 <tbody>

  <tr>

   <td width="527">% Randomly initialize the weights to small values epsiloninit = 0.12;W = rand(L out, 1 + L in) * <a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> * epsiloninit − epsiloninit;</td>

  </tr>

 </tbody>

</table>

<em>You do not need to submit any code for this part of the exercise.</em>

<h2>2.3        Backpropagation</h2>

Figure 3: Backpropagation Updates.

Now, you will implement the backpropagation algorithm. Recall that the intuition behind the backpropagation algorithm is as follows. Given a training example (<em>x</em><sup>(<em>t</em>)</sup><em>,y</em><sup>(<em>t</em>)</sup>), we will first run a “forward pass” to compute all the activations throughout the network, including the output value of the hypothesis <em>h</em><sub>Θ</sub>(<em>x</em>). Then, for each node <em>j </em>in layer <em>l</em>, we would like to compute an “error term”  that measures how much that node was “responsible” for any errors in our output.

For an output node, we can directly measure the difference between the network’s activation and the true target value, and use that to define <em>δ<sub>j</sub></em><sup>(3)</sup>

(since layer 3 is the output layer). For the hidden units, you will compute  based on a weighted average of the error terms of the nodes in layer (<em>l </em>+ 1).

In detail, here is the backpropagation algorithm (also depicted in Figure 3). You should implement steps 1 to 4 in a loop that processes one example at a time. Concretely, you should implement a for-loop for t = 1:m and place steps 1-4 below inside the for-loop, with the <em>t<sup>th </sup></em>iteration performing the calculation on the <em>t<sup>th </sup></em>training example (<em>x</em><sup>(<em>t</em>)</sup><em>,y</em><sup>(<em>t</em>)</sup>). Step 5 will divide the accumulated gradients by <em>m </em>to obtain the gradients for the neural network cost function.

<ol>

 <li>Set the input layer’s values (<em>a</em><sup>(1)</sup>) to the <em>t</em>-th training example <em>x</em><sup>(<em>t</em>)</sup>.</li>

</ol>

Perform a feedforward pass (Figure 2), computing the activations (<em>z</em><sup>(2)</sup><em>,a</em><sup>(2)</sup><em>,z</em><sup>(3)</sup><em>,a</em><sup>(3)</sup>) for layers 2 and 3. Note that you need to add a +1 term to ensure that the vectors of activations for layers <em>a</em><sup>(1) </sup>and <em>a</em><sup>(2) </sup>also include the bias unit. In Octave/MATLAB, if a 1 is a column vector, adding one corresponds to a 1 = [1 ; a 1].

<ol start="2">

 <li>For each output unit <em>k </em>in layer 3 (the output layer), set</li>

</ol>

<em>,</em>

where <em>y<sub>k </sub></em>∈ {0<em>,</em>1} indicates whether the current training example belongs to class <em>k </em>(<em>y<sub>k </sub></em>= 1), or if it belongs to a different class (<em>y<sub>k </sub></em>= 0). You may find logical arrays helpful for this task (explained in the previous programming exercise).

<ol start="3">

 <li>For the hidden layer <em>l </em>= 2, set</li>

 <li>Accumulate the gradient from this example using the following formula. Note that you should skip or remove. In Octave/MATLAB, removing corresponds to delta 2 = delta 2(2:end).</li>

</ol>

∆(<em>l</em>) = ∆(<em>l</em>) + <em>δ</em>(<em>l</em>+1)(<em>a</em>(<em>l</em>))<em>T</em>

<ol start="5">

 <li>Obtain the (unregularized) gradient for the neural network cost function by dividing the accumulated gradients by:</li>

</ol>

<strong>Octave/MATLAB Tip: </strong>You should implement the backpropagation algorithm only after you have successfully completed the feedforward and cost functions. While implementing the backpropagation algorithm, it is often useful to use the size function to print out the sizes of the variables you are working with if you run into dimension mismatch errors (“nonconformant arguments” errors in Octave/MATLAB).

After you have implemented the backpropagation algorithm, the script ex4.m will proceed to run gradient checking on your implementation. The gradient check will allow you to increase your confidence that your code is computing the gradients correctly.

<h2>2.4        Gradient checking</h2>

In your neural network, you are minimizing the cost function <em>J</em>(Θ). To perform gradient checking on your parameters, you can imagine “unrolling” the parameters Θ<sup>(1)</sup><em>,</em>Θ<sup>(2) </sup>into a long vector <em>θ</em>. By doing so, you can think of the cost function being <em>J</em>(<em>θ</em>) instead and use the following gradient checking procedure.

Suppose you have a function <em>f<sub>i</sub></em>(<em>θ</em>) that purportedly computes ); you’d like to check if <em>f<sub>i </sub></em>is outputting correct derivative values.

Let              and

So, <em>θ</em><sup>(<em>i</em>+) </sup>is the same as <em>θ</em>, except its <em>i</em>-th element has been incremented by . Similarly, <em>θ</em><sup>(<em>i</em>−) </sup>is the corresponding vector with the <em>i</em>-th element decreased by . You can now numerically verify <em>f<sub>i</sub></em>(<em>θ</em>)’s correctness by checking, for each <em>i</em>, that:

<em>.</em>

The degree to which these two values should approximate each other will depend on the details of <em>J</em>. But assuming, you’ll usually find that the left- and right-hand sides of the above will agree to at least 4 significant digits (and often many more).

We have implemented the function to compute the numerical gradient for you in computeNumericalGradient.m. While you are not required to modify the file, we highly encourage you to take a look at the code to understand how it works.

In the next step of ex4.m, it will run the provided function checkNNGradients.m which will create a small neural network and dataset that will be used for checking your gradients. If your backpropagation implementation is correct,

you should see a relative difference that is less than 1e-9.

<strong>Practical Tip: </strong>When performing gradient checking, it is much more efficient to use a small neural network with a relatively small number of input units and hidden units, thus having a relatively small number of parameters. Each dimension of <em>θ </em>requires two evaluations of the cost function and this can be expensive. In the function checkNNGradients, our code creates a small random model and dataset which is used with computeNumericalGradient for gradient checking. Furthermore, after you are confident that your gradient computations are correct, you should turn off gradient checking before running your learning algorithm.

<strong>Practical Tip: </strong>Gradient checking works for any function where you are computing the cost and the gradient. Concretely, you can use the same computeNumericalGradient.m function to check if your gradient implementations for the other exercises are correct too (e.g., logistic regression’s cost function).

<em>Once your cost function passes the gradient check for the (unregularized) neural network cost function, you should submit the neural network gradient function (backpropagation).</em>

<h2>2.5        Regularized Neural Networks</h2>

After you have successfully implemeted the backpropagation algorithm, you will add regularization to the gradient. To account for regularization, it turns out that you can add this as an additional term <em>after </em>computing the gradients using backpropagation.

Specifically, after you have computed ∆<sup>(</sup><em><sub>ij</sub><sup>l</sup></em><sup>) </sup>using backpropagation, you should add regularization using

for <em>j </em>= 0

for <em>j </em>≥ 1

Note that you should <em>not </em>be regularizing the first column of Θ<sup>(<em>l</em>) </sup>which is used for the bias term. Furthermore, in the parameters Θ is indexed starting from 1, and <em>j </em>is indexed starting from 0. Thus,

 (<em>i</em>)             (<em>l</em>)             

Θ1<em>,</em>0      Θ1<em>,</em>1        <em>…</em>

Θ(<em>l</em>) = Θ(<sub>2</sub><em>i</em><em><sub>,</sub></em>)<sub>0    </sub>Θ(<sub>2</sub><em>l</em><em><sub>,</sub></em>)<sub>1            </sub><em>.</em>

 …                  …

Somewhat confusingly, indexing in Octave/MATLAB starts from 1 (for both <em>i </em>and <em>j</em>), thus Theta1(2, 1) actually corresponds to Θ (i.e., the entry in the second row, first column of the matrix Θ<sup>(1) </sup>shown above)

Now modify your code that computes grad in nnCostFunction to account for regularization. After you are done, the ex4.m script will proceed to run gradient checking on your implementation. If your code is correct, you should expect to see a relative difference that is less than 1e-9. <em>You should now submit your solutions.</em>

<h2>2.6        Learning parameters using fmincg</h2>

After you have successfully implemented the neural network cost function and gradient computation, the next step of the ex4.m script will use fmincg to learn a good set parameters.

After the training completes, the ex4.m script will proceed to report the training accuracy of your classifier by computing the percentage of examples it got correct. If your implementation is correct, you should see a reported training accuracy of about 95.3% (this may vary by about 1% due to the random initialization). It is possible to get higher training accuracies by training the neural network for more iterations. We encourage you to try training the neural network for more iterations (e.g., set MaxIter to 400) and also vary the regularization parameter <em>λ</em>. With the right learning settings, it is possible to get the neural network to perfectly fit the training set.

<h1>3          Visualizing the hidden layer</h1>

One way to understand what your neural network is learning is to visualize what the representations captured by the hidden units. Informally, given a particular hidden unit, one way to visualize what it computes is to find an input x that will cause it to activate (that is, to have an activation value

) close to 1). For the neural network you trained, notice that the <em>i<sup>th </sup></em>row of Θ<sup>(1) </sup>is a 401-dimensional vector that represents the parameter for the <em>i<sup>th </sup></em>hidden unit. If we discard the bias term, we get a 400 dimensional vector that represents the weights from each input pixel to the hidden unit.

Thus, one way to visualize the “representation” captured by the hidden unit is to reshape this 400 dimensional vector into a 20 × 20 image and display it.<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> The next step of ex4.m does this by using the displayData function and it will show you an image (similar to Figure 4) with 25 units, each corresponding to one hidden unit in the network.

In your trained network, you should find that the hidden units corresponds roughly to detectors that look for strokes and other patterns in the input.

Figure 4: Visualization of Hidden Units.

<h2>3.1        Optional (ungraded) exercise</h2>

In this part of the exercise, you will get to try out different learning settings for the neural network to see how the performance of the neural network varies with the regularization parameter <em>λ </em>and number of training steps (the

MaxIter option when using fmincg).

Neural networks are very powerful models that can form highly complex decision boundaries. Without regularization, it is possible for a neural network to “overfit” a training set so that it obtains close to 100% accuracy on the training set but does not as well on new examples that it has not seen before. You can set the regularization <em>λ </em>to a smaller value and the MaxIter parameter to a higher number of iterations to see this for youself.

You will also be able to see for yourself the changes in the visualizations of the hidden units when you change the learning parameters <em>λ </em>and MaxIter.

<em>You do not need to submit any solutions for this optional (ungraded) exercise.</em>

<h1>Submission and Grading</h1>

After completing various parts of the assignment, be sure to use the submit function system to submit your solutions to our servers. The following is a breakdown of how each part of this exercise is scored.

<table width="504">

 <tbody>

  <tr>

   <td width="262"><strong>Part</strong></td>

   <td width="156"><strong>Submitted File</strong></td>

   <td width="87"><strong>Points</strong></td>

  </tr>

  <tr>

   <td width="262">Feedforward and Cost Function</td>

   <td width="156">nnCostFunction.m</td>

   <td width="87">30 points</td>

  </tr>

  <tr>

   <td width="262">Regularized Cost Function</td>

   <td width="156">nnCostFunction.m</td>

   <td width="87">15 points</td>

  </tr>

  <tr>

   <td width="262">Sigmoid Gradient</td>

   <td width="156">sigmoidGradient.m</td>

   <td width="87">5 points</td>

  </tr>

  <tr>

   <td width="262">Neural    Net     Gradient     Function(Backpropagation)</td>

   <td width="156">nnCostFunction.m</td>

   <td width="87">40 points</td>

  </tr>

  <tr>

   <td width="262">Regularized Gradient</td>

   <td width="156">nnCostFunction.m</td>

   <td width="87">10 points</td>

  </tr>

  <tr>

   <td width="262">Total Points</td>

   <td width="156"> </td>

   <td width="87">100 points</td>

  </tr>

 </tbody>

</table>

You are allowed to submit your solutions multiple times, and we will take only the highest score into consideration.

<a href="#_ftnref1" name="_ftn1">[1]</a> Octave is a free alternative to MATLAB. For the programming exercises, you are free to use either Octave or MATLAB.

<a href="#_ftnref2" name="_ftn2">[2]</a> One effective strategy for choosing is to base it on the number of units in the network. A good choice of, where <em>L<sub>in </sub></em>= <em>s<sub>l </sub></em>and <em>L<sub>out </sub></em>= <em>s<sub>l</sub></em><sub>+1 </sub>are

the number of units in the layers adjacent to Θ<sup>(<em>l</em>)</sup>.

<a href="#_ftnref3" name="_ftn3">[3]</a> It turns out that this is equivalent to finding the input that gives the highest activation for the hidden unit, given a “norm” constraint on the input (i.e., k<em>x</em>k<sub>2 </sub>≤ 1).<img decoding="async" data-recalc-dims="1" data-src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2022/04/880.png?w=980&amp;ssl=1" class="lazyload" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

 <noscript>

  <img decoding="async" src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2022/04/880.png?w=980&amp;ssl=1" data-recalc-dims="1">

 </noscript>