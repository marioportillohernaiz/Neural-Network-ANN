import java.io.*;

public class coursework {
    public static void main(String[] args) {
        int inputNodes = 5;
        int hiddenNodes = 10;
        int firstHiddenNode = inputNodes + 1;
        int totalNumbNodes = inputNodes + hiddenNodes + 1;
        int indexOuputNode = firstHiddenNode + hiddenNodes + 1;
        double p = 0.1;
        String fileName = "C:\\Users\\porti\\Desktop\\AI CW\\TrainingStandarisedData.csv";
        String outputFileName = "C:\\Users\\porti\\Desktop\\AI CW\\outputNode.csv";
        String meanFileName = "C:\\Users\\porti\\Desktop\\AI CW\\mean.csv";
        
        // Numb of Epochs
        int numbOfEpoch = 70;

        double[][] fileRead = createArray(fileName);
        int numbOfDataRows = getNumbOfDataRows(fileName);
        double[] outputNodes = new double[numbOfDataRows];
        double[] mean = new double[numbOfEpoch+1];


        double[] nodeValues = new double[totalNumbNodes + 1];
        double[] deltas = new double[totalNumbNodes + 1];

        Double[][] weightAndBias = createWandB(inputNodes, hiddenNodes);


        for (int epoch = 0; epoch <= numbOfEpoch; epoch++) { 
            for (int i = 0; i < numbOfDataRows; i++) {
                for (int update=1; update<inputNodes+1; update++) {
                    weightAndBias[0][update] = fileRead[i][update-1];
                }


                // for (int row = 0; row < weightAndBias.length; row++) {
                //     for (int col = 0; col < weightAndBias[row].length; col++) {
                //         System.out.print(weightAndBias[row][col] + ", ");
                //     }
                //     System.out.println();
                // }

                // FORWARD PASS
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


                // BACKWARD PASS
                double u5 = nodeValues[indexOuputNode-1];
                double delta5 = (fileRead[i][5] - u5) * (u5 * (1 - u5));

                for (int hiddenN = firstHiddenNode; hiddenN < indexOuputNode - 1; hiddenN++) {
                    double u = nodeValues[hiddenN];
                    deltas[hiddenN] = (weightAndBias[hiddenN][indexOuputNode - 1] * (delta5)) * (u * (1 - u));
                }

                // Updating weights
                for (int hiddenN = firstHiddenNode; hiddenN < indexOuputNode; hiddenN++) {
                    if (hiddenN != indexOuputNode - 1) {

                        // Updating bias & weights for hidden nodes
                        for (int inputN = 0; inputN < firstHiddenNode; inputN++) { // 0-5
                            weightAndBias[inputN][hiddenN] = weightAndBias[inputN][hiddenN] + p * (deltas[hiddenN]) * (weightAndBias[0][inputN]);
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
                sum += Math.pow((fileRead[i][5] - outputNodes[i]), 2);
            }
            // System.out.println("Mean: " + (sum/numbOfDataRows));
            mean[epoch] = sum/numbOfDataRows;
        }

        // Writes the output node of the last epoch in a file
        writeToFile(outputFileName, outputNodes, numbOfDataRows);

        // Writes the mean into a file
        writeToFile(meanFileName, mean, numbOfEpoch);
    }

    public static double[][] createArray(String fileName) {
        String line = "";
        int numbOfRows = 0;
        double[][] returnArr = new double[1][1];

        // Read the number of lines in the file
        try {
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            while ((line = br.readLine()) != null) {
                numbOfRows++;
            }
            returnArr = new double[numbOfRows][6];
            br.close();

        } catch (IOException io) {
            System.out.println(io);
        }

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
}