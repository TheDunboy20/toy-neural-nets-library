package msmarik.activations.functions;

import msmarik.activations.Activation;
import msmarik.activations.DoubleArrayUnaryOperator;

public class Softmax implements Activation {
    @Override
    public DoubleArrayUnaryOperator standard() {
        return arr -> {
            double rollingSum = 0;
            for (double element : arr) {
                rollingSum += Math.exp(element);
            }
            final double finalSum = rollingSum;
            double[] result = new double[arr.length];

            for (int i = 0; i < arr.length; i++) {
                result[i] = Math.exp(arr[i]) / finalSum;
            }
            return result;
        };
    }

    @Override
    public DoubleArrayUnaryOperator derivative() {
        throw new UnsupportedOperationException("Softmax is a Jacobian matrix not a vector");
    }
}
