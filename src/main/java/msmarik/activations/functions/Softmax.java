package msmarik.activations.functions;

import msmarik.activations.Activation;
import msmarik.activations.DoubleArrayUnaryOperator;

import java.util.function.DoubleUnaryOperator;

public class Softmax implements Activation {
    @Override
    public DoubleArrayUnaryOperator standard() {
        return arr -> {
            double rollingSum = 0;
            for (double element : arr) {
                rollingSum += Math.exp(element);
            }
            final double finalSum = rollingSum;

            return elementwise(x -> (Math.exp(x)));
        };
    }

    @Override
    public DoubleUnaryOperator derivative() {
        return null;
    }
}
