import java.io.*;

public class improvements {
    public static void main(String[] args) {
        int inputNodes = 5;
        int hiddenNodes = 10;
        int firstHiddenNode = inputNodes + 1;
        int totalNumbNodes = inputNodes + hiddenNodes + 1;
        int indexOuputNode = firstHiddenNode + hiddenNodes + 1;
        double p = 0.1;
        String fileName = "C:\\Users\\porti\\Desktop\\AI CW\\TestStandarisedData.csv"; // Change the file path
        String validationFile = "C:\\Users\\porti\\Desktop\\AI CW\\ValidationStandarisedData.csv";
        String outputFileName = "C:\\Users\\porti\\Desktop\\AI CW\\outputNode.csv";
        String meanFileName = "C:\\Users\\porti\\Desktop\\AI CW\\mean.csv";
        String validationMean = "C:\\Users\\porti\\Desktop\\AI CW\\validationMean.csv";
        double momentumVal = 0;

        boolean momentum = true;
        boolean annealing = true;
        boolean boldDriver = true;
        boolean weightDecay = false;

        double oldMean = 0;
        
        // Numb of Epochs
        int numbOfEpoch = 100;

        // Training set variables
        double[][] fileRead = createArray(fileName);
        int numbOfDataRows = getNumbOfDataRows(fileName);
        double[] TrainingOutputNodes = new double[numbOfDataRows];
        double[] mean = new double[numbOfEpoch+1];

        // Validation set variables
        double[][] ValidatingArr = createArray(validationFile);
        int numbOfValRows = getNumbOfDataRows(validationFile);
        double[] ValidatingOutputNodes = new double[numbOfValRows];
        double[] valMean = new double[numbOfEpoch+1];
        
        
        double[] nodeValues = new double[totalNumbNodes + 1];
        double[] deltas = new double[totalNumbNodes + 1];

        // Creating the two-dimension array
        Double[][] weightAndBias = createWandB(inputNodes, hiddenNodes);

        // TRAINING 
        for (int epoch = 0; epoch <= numbOfEpoch; epoch++) { // Looping through epochs
            Double[][] prevWeightAndBias = weightAndBias;

            for (int i = 0; i < numbOfDataRows; i++) { // Looping through each data row
                // Update inputs from the dataset
                for (int update=1; update<inputNodes+1; update++) {
                    weightAndBias[0][update] = fileRead[i][update-1];
                }

                // FORWARD PASS
                forwardPass(i, firstHiddenNode, indexOuputNode, weightAndBias, nodeValues, TrainingOutputNodes);

                // BACKWARD PASS
                double delta5 = calculateDeltas(i, indexOuputNode, firstHiddenNode, weightAndBias, nodeValues, fileRead, deltas, epoch, weightDecay, p);


                // Updating weights
                for (int hiddenN = firstHiddenNode; hiddenN < indexOuputNode; hiddenN++) {
                    if (hiddenN != indexOuputNode - 1) {

                        // Updating bias & weights for hidden nodes
                        for (int inputN = 0; inputN < firstHiddenNode; inputN++) {
                            // Adding momentum
                            if (momentum == true && epoch != 0) {
                                momentumVal = 0.9 * (p * (deltas[hiddenN]) * (weightAndBias[0][inputN]));
                            }
                            weightAndBias[inputN][hiddenN] = weightAndBias[inputN][hiddenN] + (p * (deltas[hiddenN]) * (weightAndBias[0][inputN])) + momentumVal;
                        }
                    } else {

                        // Updating bias for output node
                        weightAndBias[0][hiddenN] = weightAndBias[0][hiddenN] + p * (delta5);
                        
                        for (int node = firstHiddenNode; node < indexOuputNode - 1; node++) {
                            weightAndBias[node][hiddenN] = weightAndBias[node][hiddenN] + p * (delta5) * (nodeValues[node]);
                        }
                    }
                }
            }


            // Calculates mean
            double sum = 0;
            for (int i=0; i<numbOfDataRows; i++) {
                sum += Math.pow((fileRead[i][5] - TrainingOutputNodes[i]), 2);
            }
            mean[epoch] = sum/numbOfDataRows;

            // Calculates Annealing
            if (annealing == true) {
                p = annealingCalc(numbOfEpoch, epoch, p);
            }

            // Updates p within the Bold Driver laws
            if (epoch % 100 == 0 && epoch != 0 && boldDriver == true) {
                double percentageIncrease = ((oldMean - mean[epoch]) / oldMean) * 100;

                if (p >= 0.01 && p <= 0.5) {
                    if (percentageIncrease >= 4) {
                        p = p * 0.7;
                        weightAndBias = prevWeightAndBias;
                    } else if (percentageIncrease < 0) {
                        p = p * 1.05;
                        weightAndBias = prevWeightAndBias;
                    }
                }
                oldMean = mean[epoch];
            } else if (epoch == 0) {
                oldMean = mean[epoch];
            }
        }

        // Writes the output node of the last epoch in a file
        writeToFile(outputFileName, TrainingOutputNodes, numbOfDataRows);

        // Writes the mean into a file
        writeToFile(meanFileName, mean, numbOfEpoch);


        // VALIDATING
        for (int j = 0; j < numbOfValRows; j++) {
            for (int update=1; update<inputNodes+1; update++) {
                weightAndBias[0][update] = ValidatingArr[j][update-1];
            }

            // FORWARD PASS
            forwardPass(j, firstHiddenNode, indexOuputNode, weightAndBias, nodeValues, ValidatingOutputNodes);
        }
        // calculating mean for validation
        double sumVal = 0;
        for (int j=0; j<numbOfValRows; j++) {
            sumVal += Math.pow((ValidatingArr[j][5] - ValidatingOutputNodes[j]), 2);
        }
        valMean[0] = sumVal/numbOfValRows;

        // Writes the Validation mean into a file
        writeToFile(validationMean, valMean, numbOfEpoch);
    }


    public static void forwardPass(int i, int firstHiddenNode, int indexOuputNode, Double[][] weightAndBias, double[] nodeValues, double[] outputNodes) {
        for (int u = firstHiddenNode; u < indexOuputNode; u++) {
            double node = weightAndBias[0][u];

            if (u != indexOuputNode - 1) {
                for (int inputN = 1; inputN < firstHiddenNode; inputN++) {
                    node += weightAndBias[inputN][u] * weightAndBias[0][inputN];
                }
                nodeValues[u] = 1 / (1 + Math.exp(-(node)));
                
            } else { 
                for (int weight = firstHiddenNode; weight < indexOuputNode - 1; weight++) { 
                    node += weightAndBias[weight][u] * nodeValues[weight];
                }
                nodeValues[u] = 1 / (1 + Math.exp(-(node)));
                outputNodes[i] = nodeValues[u];
            }
        }
    }

    public static double calculateDeltas(int i, int indexOuputNode, int firstHiddenNode, Double[][] weightAndBias, double[] nodeValues, double[][] fileRead, double[] deltas, int epoch, boolean weightDecay, double p) {
        double u5 = nodeValues[indexOuputNode-1];
        double delta5 = 0;

        // Using weight decay as an improvement
        if (weightDecay == true) {
            double count = 0;
            double omega = 0;
            double upsilon = 0;

            for (int row = 0; row < weightAndBias.length; row++) {
                for (int col = firstHiddenNode; col < indexOuputNode; col++) { // change
                    
                    if (weightAndBias[row][col] != null && row != col) {
                        omega += Math.pow(weightAndBias[row][col], 2);
                        count++;
                    }
                }
            }
            omega = omega * (1/(2*(count)));
            if(epoch != 0) {
                upsilon = 1 / (p * epoch);
            }
    
            delta5 = ((fileRead[i][firstHiddenNode-1] - u5) + (omega * upsilon)) * (u5 * (1 - u5));
        } else {
            delta5 = (fileRead[i][firstHiddenNode-1] - u5) * (u5 * (1 - u5));
        }

        for (int hiddenN = firstHiddenNode; hiddenN < indexOuputNode - 1; hiddenN++) {
            double u = nodeValues[hiddenN];
            deltas[hiddenN] = (weightAndBias[hiddenN][indexOuputNode - 1] * (delta5)) * (u * (1 - u));
        }
        return delta5;
    }

    public static double[][] createArray(String fileName) {
        String line = "";
        double[][] returnArr = new double[1][1];

        // Read the number of lines in the file
        int numbOfRows = getNumbOfDataRows(fileName);
        returnArr = new double[numbOfRows][6];

        // Create the Array
        try {
            int row = 0;
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            while ((line = br.readLine()) != null) {
                String[] rowOfData = line.split(",");
                for (int x = 0; x < 6; x++) {
                    returnArr[row][x] = Double.parseDouble(rowOfData[x]);
                }
                row++;
            }
            br.close();
        } catch (IOException io) {
            System.out.println(io);
        }

        return returnArr;
    }

    public static int getNumbOfDataRows(String fileName) {
        int numbOfRows = 0;
        String line = "";
        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            while ((line = br.readLine()) != null) {
                numbOfRows++;
            }
            br.close();

        } catch (IOException io) {
            System.out.println(io);
        }
        return numbOfRows;
    }

    public static double getRandomNumber(double inputNodes) {
        double randomNumb = (-2 / (inputNodes)) + (Math.random() * ((2 / (inputNodes)) - (-2 / (inputNodes))));
        return randomNumb;
    }

    public static void writeToFile(String fileName, double[] array, int endOfLoop) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))) {
            for (int i = 0; i < endOfLoop; i++) {
                bw.write(Double.toString(array[i]));
                bw.newLine();
            }
        } catch (IOException io) {
            System.out.println(io);
        }
    }

    public static Double[][] createWandB(int inputNodes, int hiddenNodes) {
        int max = 1 + inputNodes + hiddenNodes + 1;
        Double[][] returnArr = new Double[max][max];
        
        for (int row=0; row<max; row++) {
            for (int col=0; col<max; col++) {
                if (row == col) {
                    returnArr[row][col] = 1.0;
                } else if(col<row) {
                    returnArr[row][col] = null;
                } else if (row == 0) {
                    if (col > inputNodes) {
                        returnArr[row][col] = getRandomNumber(inputNodes);
                    } else {
                        returnArr[row][col] = 0.0;
                    }
                } else if (col > inputNodes && col < max-1 && row != 0) {
                    if (!(row > inputNodes && row < max)) {
                        returnArr[row][col] = getRandomNumber(inputNodes);
                    }
                } else if (col == max-1 && row > inputNodes) {
                    returnArr[row][col] = getRandomNumber(inputNodes);
                }
            }
        }
        return returnArr;
    }

    public static double annealingCalc(int numbOfEpoch, int epoch, double p) {
        double endP = 0.01;
        double q = 0.1;

        p = endP + ((q - endP)*(1-(1/(1+Math.exp(10-((20*epoch)/numbOfEpoch))))));
        return p;
    }
}