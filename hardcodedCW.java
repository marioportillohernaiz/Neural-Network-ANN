// Example code from the slides
public class hardcodedCW {
    public static void main(String[] args) {
        double p = 0.1;

        double[][] weightAndBias = 
            {{1, 1, 0, 1, -6, -3.92}, 
            {0, 1, 0, 3, 6, 0}, 
            {0, 0, 1, 4, 5, 0}, 
            {0, 0, 0, 1, 0, 2}, 
            {0, 0, 0, 0, 1, 4},
            {0, 0, 0, 0, 0, 1},
        }; 
        
        // 1st row = u3 = (1,3 * 0,1) + (2,3 * 0,2) + (3,3 * 0,3)
        // {{bias}, {u1 weights}, {u2 weights}, {u3 weights}, {u4 weights}, {u5 weights}}
        // S3 = 1x3 + 0x4 + 1x1 = 4 // (1stInputCell)w + (2ndInputCell)w + u3x1


        for (int i=0; i<20000; i++) {
            // FORWARD PASS
            double u3 = 1/(1+Math.exp(-((weightAndBias[1][3]*weightAndBias[0][1]) + (weightAndBias[2][3]*weightAndBias[0][2]) + (weightAndBias[3][3]*weightAndBias[0][3]))));
            double u4 = 1/(1+Math.exp(-((weightAndBias[1][4]*weightAndBias[0][1]) + (weightAndBias[2][4]*weightAndBias[0][2]) + (weightAndBias[4][4]*weightAndBias[0][4]))));
            double u5 = 1/(1+Math.exp(-((weightAndBias[3][5]*u3) + (weightAndBias[4][5]*u4) + (weightAndBias[5][5]*weightAndBias[0][5]))));
            System.out.println("Output Node: " + u5);

            // BACKWARD PASS
            // Correct Output = 1
            // Sigma5 = (CorrectOutput - u5)( u5(1-u5) )
            // hn3 = (w3>5(Sigma5))( u5(1-u5) )

            //double delta5 = (1-u5)*(u5*(1-u5));
            // double delta3 = (weightAndBias[3][5]*(delta5))*(u3*(1-u3));
            // double delta4 = (weightAndBias[4][5]*(delta5))*(u4*(1-u4));

            double delta5 = (1-u5)*(u5*(1-u5));
            double delta3 = (weightAndBias[3][5]*(delta5))*(u3*(1-u3));
            double delta4 = (weightAndBias[4][5]*(delta5))*(u4*(1-u4));

            // Bias on node 3:
            double w03 = weightAndBias[0][3] + p*(delta3)*(weightAndBias[0][1]);   // bias 3 (so weight 1)
            double w13 = weightAndBias[1][3] + p*(delta3)*(weightAndBias[0][1]); // input 1 to bias 3
            double w23 = weightAndBias[2][3] + p*(delta3)*(weightAndBias[0][2]); // input 2 to bias 3
            // System.out.println("w03: "+ w03);
            // System.out.println("w13: "+ w13);
            // System.out.println("w23: "+ w23);
            
            // Bias on node 4:
            double w04 = weightAndBias[0][4] + p*(delta4)*(weightAndBias[0][1]);
            double w14 = weightAndBias[1][4] + p*(delta4)*(weightAndBias[0][1]);
            double w24 = weightAndBias[2][4] + p*(delta4)*(weightAndBias[0][2]);
            // System.out.println("w04: "+ w04);
            // System.out.println("w14: "+ w14);
            // System.out.println("w24: "+ w24);
            
            // Bias on node 5:
            double w05 = weightAndBias[0][5] + p*(delta5)*(weightAndBias[0][1]);
            double w35 = weightAndBias[3][5] + p*(delta5)*(u3);
            double w45 = weightAndBias[4][5] + p*(delta5)*(u4);
            // System.out.println("w05: "+ w05);
            // System.out.println("w35: "+ w35);
            // System.out.println("w45: "+ w45);


            // bias 3, 4, 5
            weightAndBias[0][3] = w03;
            weightAndBias[0][4] = w04;
            weightAndBias[0][5] = w05;

            //weights input1
            weightAndBias[1][3] = w13;
            weightAndBias[1][4] = w14;
            //weights input2
            weightAndBias[2][3] = w23;
            weightAndBias[2][4] = w24;
            //weight bias3
            weightAndBias[3][5] = w35;
            //weight bias4
            weightAndBias[4][5] = w45;
        }
    }
}