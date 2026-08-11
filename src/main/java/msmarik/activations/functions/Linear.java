package msmarik.activations.functions;

import msmarik.activations.Activation;
import msmarik.activations.DoubleArrayUnaryOperator;

import java.util.Arrays;

public record Linear() implements Activation {
    @Override
    public DoubleArrayUnaryOperator standard() {
        return arr -> arr;
    }

    // TODO: Once enough data is present, performance of this function vs using streams might be compared.
    @Override
    public DoubleArrayUnaryOperator derivative() {
        return arr -> {
            double[] result = new double[arr.length];
            Arrays.fill(result, 1);
            return result;
        };
    }
}
