# Neural-Network-ANN

<p>The following repository holds an Artificial Neural Network - specifically a Multi-Layer Perceptron (MLP) - trained using Backpropagation. I will be describing how I pre-processed data, implemented the algorithm, trained and selected the network, evaluate the model and compared it with another model.</p>

<h2>Data pre-processing</h2>
<p>In this section I will describe how I will clean the original dataset such as removing any outliers or incorrect values stored in the dataset and standardising them. The first step was to search each column on the original dataset and find any incorrect formatted values such as letters or empty spaces:</p>
<img style="width: 300px; height: auto" src="https://user-images.githubusercontent.com/111706273/227553191-abaa6d57-d9e3-4cd2-99b5-c08a50fa054a.png">
<p>The way these rows where selected was using “formatting” tools in the Excell Data Sheet. After removing a large number of incorrect formatted values or outliers (such as the temperature being 180°C – which is unreasonable) I used the z-score method to remove any other outliers. The formula used is:</p>
<p><strong> z=(x-mean)/std_deviation</strong>.</p>
<p>Which calculates the distance and mean in standard deviation units. A value greater than a threshold is considered an outlier, such as 3 standard deviations from the mean.</p>
<p>After removing all the outliers, we can visualize cleaner graphs, such as:</p>
<img style="width: 300px; height: auto" src="https://user-images.githubusercontent.com/111706273/227554005-cb6cd8b5-180e-48cc-897c-054a86bbbe4d.png">
<p>Our whole dataset would now look like this:</p>
<img style="width: 300px; height: auto" src="https://user-images.githubusercontent.com/111706273/227554080-418c3d74-5ed2-49ed-8e16-d328d1e1768f.png">
<p>The next step was to split the dataset into three sets; the Training set (which consisted of 60% of the data), the Validation set (consisting of 20% of the data) and the Test set (consisting of the last 20%). The Training set will be used to create and work on the model, as well as the Validation set. In addition, the Validation set will be used to pick the ‘best’ model after it has been trained. The Test set will then be used for the final test on our ‘best’ model.
The way I split the data was using “Ablebit Tools” which can be downloaded for Microsoft Excel, which can select a random number of rows depending on the percentage you choose.
Once the data was split in the three sets (the column “Date” was removed from each split), we can standardize it. The equation used to standardize the data is as follows, using the max and min of each set and each column, we can calculate the standardization of each value (Ri) using the following equation.</p>
<p><strong>S_i=(R_i - Min)/(Max - Min)</strong>.</p>

<h2>Implementation of Algorithm</h2>
<p>The first step to implementing my Multi-Layer Perceptron was to hard-code the example given in lectures to make it easier to understand how an MLP worked. I was able to create the MLP with a fixed number of 2 inputs and 2 hidden nodes and recreate the correct output given in the lectures with both, one epoch and 20,000 epochs.
The next step was to automate the MLP so that it was possible to add multiple inputs and hidden nodes. The diagram, bellow represents how the MLP is structured. The number of inputs in the artificial network will be equal to the length of the dataset. The number of hidden nodes can be updated in order to obtain the most optimal MLP. Finally, there will be only one output node which we will try to approximate it to the PanE after each epoch.</p>
<img style="width: 20px; height: auto" src="https://user-images.githubusercontent.com/111706273/227556304-054700ab-b40d-47b0-a019-b9183ed91f9d.png">
<p>My implementation of the MLP has been made using Java, where it consists of one class and multiple methods all of which are used for the neural network, including X improvements I applied to my MLP, which I will discuss later on.
The main() method starts by setting all the variables I will need to store and run the Neural Network as well as call other methods. Some of these variables call the following methods:</p>
<ul>
<li>createArray() (double[][] type) which passes the “filename” string argument. This method reads the CVS file containing the Data Set (either Training, Validating or Testing set) and stores it into a two-dimension array.</li>
<li>getNumbOfDataRows() (int type) which passes the “filename” string argument. This method reads the CSV file containing the data set and counts the number of rows in the file (the number of rows of data).</li>
<li>createWandB() (Double[][] type) which passes “inputNodes” and “hiddenNodes” both as int arguments. The following method creates the two-dimension array in which the Neural Network, meaning the values of the Biases and the Weights, is stored. The first row contains the Biases of the MLP such as WandB[0][1] being the first input (which will be empty for now). WandB[1][6] for example, will then be the weight between Bias 1 and Bias 6, if there is no connection between biases then the value will be “null” otherwise it will be random using the following method, unless it’s an input.</li>
<li>getRandomNumber() (double type) which passes the “inputNodes” int argument. This method creates a random number with range -2/n≤x≤2/(n ) where n is the number of input nodes.</li>
</ul>

<p>The next step in the main() method is to run through the algorithm that calculates the output node and updates the weights. To do this I run through a nested “for” loop where the first loop loops through the number of epochs and the second loop loops through the number of rows in the dataset. 
In the loop, for each row, I firstly add the new inputs to the weight-and-Bias two-dimension array. Then I call the forwardPass() method which passes various arguments in order to calculate the forward pass of our MLP. For each row, we calculate the forward pass using the following equations.</p>

<p><strong>S_i=∑〖u_i * w_(i,j)〗   and   f(S_i) = 1/(1 + e^(-S_i))</strong>.</p>

<p>The values calculated for each bias will be stored in a “double[]” array, called “nodeValues[]”, for each forward pass and replaced on the next forward pass. On the other hand, for each row, their output node will be calculated and stored in a different array in order to keep track of them and therefore make it easier for us to calculate the Mean Square Error.</p>

<p>The next step is to calculate our delta values which I will do so through another method called calculateDeltas(). Similar to “forwardPass()”, this method passes various arguments in order to calculate the new deltas. It will also return the delta of the output node as a double in order to apply it to our updating weights. In this method, the deltas of each hidden node will be calculated and saved in another “double[]” array so that we are able to loop through them when updating our weights. The equations used to calculate the deltas for the hidden nodes are as follows.</p>
<p><strong>δ_i = (w_(i,j) * δ_(i+1))*(u_i * (1 - u_i))</strong>.</p>
<p>The equation used to calculate the delta for the output node varies a bit:</p>
<p><strong>δ_i = (C_i - u_i)*(u_i * (1 - u_i))</strong>.</p>
<p>Once we have calculated our delta values, we can move onto the final steps. Updating the weights, calculating the mean and writing our data to our files. 
When updating the weights, I move backwards from the output node to the first input, replacing all values on the way. My mean values are then calculated after each epoch using the following equation. Last step is to write all these values to a CSV file using the writeToFile() method so that we can display them on graphs. The method to write to the file is through a buffer reader.</p>


<h2>Training and network selection</h2>
<p>After training my neural network, I recorded the MSE after 70 epochs each time I tested it. The reason I used this number of epochs is due to the MSE being very unnoticeable after more than 70 epochs, which makes it hard to visualize. Without any improvements, firstly I tested my MLP with two hidden nodes and compared it to data from a 5 and 10 hidden nodes MLP.</p>
<img style="width: 300px; height: auto" src="https://user-images.githubusercontent.com/111706273/227566198-01b9433b-648e-4d57-b014-a8104e8599a8.png">
<p>As shown in the graph above, the number of hidden nodes have an impact on our neural network. So as we can see, having more hidden nodes in the neural network improves how fast the MSE is lowered. Therefore, the most optimal number of nodes for my MLP is to have 10 hidden nodes for my final model.</p>

<h3>Momentum Improvement</h3>
<img style="width: 200px; height: auto" align="left" src="https://user-images.githubusercontent.com/111706273/227570720-d9a49f38-0944-4898-866d-bdecfd1ab578.png">
<p align="left">By implementing momentum, we can see that our model has improved drastically on our MSE values. With momentum, our MSE reaches close to zero around the 15th epoch whereas without momentum, our MSE reaches close to zero around the 30-epoch mark.</p>
<br><br>

```
// Updating bias & weights for hidden nodes
for (int inputN = 0; inputN < firstHiddenNode; inputN++) {
  // Adding momentum
  if (momentum == true && epoch != 0) {
    momentumVal = 0.9 * (p * (deltas[hiddenN]) * (weightAndBias[0][inputN]));
  }
  weightAndBias[inputN][hiddenN] = weightAndBias[inputN][hiddenN] + (p * (deltas[hiddenN]) * (weightAndBias[0][inputN])) + momentumVal;
}
```
