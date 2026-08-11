package msmarik.activations;

@FunctionalInterface
public interface DoubleArrayUnaryOperator {
    double[] applyAsDoubleArray(double[] values);
}
