# Neural-Network-ANN

<p>The following repository holds an Artificial Neural Network - specifically a Multi-Layer Perceptron (MLP) - trained using Backpropagation. I will be describing how I pre-processed data, implemented the algorithm, trained and selected the network, evaluate the model and compared it with another model.</p>

<h2>Data pre-processing</h2>
<p>In this section I will describe how I will clean the original dataset such as removing any outliers or incorrect values stored in the dataset and standardising them. The first step was to search each column on the original dataset and find any incorrect formatted values such as letters or empty spaces:</p>
![image](https://user-images.githubusercontent.com/111706273/227553191-abaa6d57-d9e3-4cd2-99b5-c08a50fa054a.png)
<p>The way these rows where selected was using “formatting” tools in the Excell Data Sheet. After removing a large number of incorrect formatted values or outliers (such as the temperature being 180°C – which is unreasonable) I used the z-score method to remove any other outliers. The formula used is: <strong> z=(x-mean)/std_deviation</strong>. Which calculates the distance and mean in standard deviation units. A value greater than a threshold is considered an outlier, such as 3 standard deviations from the mean. The following data rows are some of these outliers I removed from the dataset.</p>
