# Neural-Network-ANN

<p>The following repository holds an Artificial Neural Network - specifically a Multi-Layer Perceptron (MLP) - trained using Backpropagation. I will be describing how I pre-processed data, implemented the algorithm, trained and selected the network, evaluate the model and compared it with another model.</p>

<h2>Data pre-processing</h2>
<p>In this section I will describe how I will clean the original dataset such as removing any outliers or incorrect values stored in the dataset and standardising them. The first step was to search each column on the original dataset and find any incorrect formatted values such as letters or empty spaces:</p>
![image](https://user-images.githubusercontent.com/111706273/227553191-abaa6d57-d9e3-4cd2-99b5-c08a50fa054a.png)
<p>The way these rows where selected was using “formatting” tools in the Excell Data Sheet. After removing a large number of incorrect formatted values or outliers (such as the temperature being 180°C – which is unreasonable) I used the z-score method to remove any other outliers. The formula used is:</p>
<p><strong> z=(x-mean)/std_deviation</strong>.</p>
<p>Which calculates the distance and mean in standard deviation units. A value greater than a threshold is considered an outlier, such as 3 standard deviations from the mean.</p>
<p>After removing all the outliers, we can visualize cleaner graphs, such as:</p>
![image](https://user-images.githubusercontent.com/111706273/227554005-cb6cd8b5-180e-48cc-897c-054a86bbbe4d.png)
<p>Our whole dataset would now look like this:</p>
![image](https://user-images.githubusercontent.com/111706273/227554080-418c3d74-5ed2-49ed-8e16-d328d1e1768f.png)
<p>The next step was to split the dataset into three sets; the Training set (which consisted of 60% of the data), the Validation set (consisting of 20% of the data) and the Test set (consisting of the last 20%). The Training set will be used to create and work on the model, as well as the Validation set. In addition, the Validation set will be used to pick the ‘best’ model after it has been trained. The Test set will then be used for the final test on our ‘best’ model.
The way I split the data was using “Ablebit Tools” which can be downloaded for Microsoft Excel, which can select a random number of rows depending on the percentage you choose.
Once the data was split in the three sets (the column “Date” was removed from each split), we can standardize it. The equation used to standardize the data is as follows, using the max and min of each set and each column, we can calculate the standardization of each value (Ri) using the following equation.</p>
<p><strong>S_i=(R_i - Min)/(Max - Min)</strong>.</p>

