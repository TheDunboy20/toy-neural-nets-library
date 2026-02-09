package msmarik.activations;

import java.util.function.DoubleUnaryOperator;

public interface Activation {
    DoubleUnaryOperator standard();
    DoubleUnaryOperator derivative();
}
