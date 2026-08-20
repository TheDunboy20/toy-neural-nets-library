package msmarik.activations.functions;

import msmarik.activations.Activation;
import msmarik.activations.DoubleArrayUnaryOperator;

public record Tanh() implements Activation {

    @Override
    public DoubleArrayUnaryOperator standard() {
        return elementwise(Math::tanh);
    }

    @Override
    public DoubleArrayUnaryOperator derivative() {
        return elementwise(x -> 1 - Math.pow(Math.tanh(x), 2));
    }
}
