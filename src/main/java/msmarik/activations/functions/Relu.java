package msmarik.activations.functions;

import msmarik.activations.Activation;
import msmarik.activations.DoubleArrayUnaryOperator;

import java.util.function.DoubleUnaryOperator;

public record Relu() implements Activation {

    @Override
    public DoubleArrayUnaryOperator standard() {
        return elementwise(x -> Math.max(0, x));
    }

    @Override
    public DoubleArrayUnaryOperator derivative() {
        return elementwise(x -> (x > 0 ? 1 : 0));
    }
}
