package msmarik.activations;

import java.util.function.DoubleUnaryOperator;

public interface Activation {
    DoubleArrayUnaryOperator standard();
    DoubleArrayUnaryOperator derivative();

    default DoubleArrayUnaryOperator elementwise(DoubleUnaryOperator operator) {
        return arr -> {
            double[] result = new double[arr.length];

            for (int i = 0; i < arr.length; i++) {
                result[i] = operator.applyAsDouble(arr[i]);
            }
            return result;
        };
    }
}
