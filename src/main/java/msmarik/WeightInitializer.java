package msmarik;

public interface WeightInitializer {
    default double initializeWeight(int inputSize, int outputSize) {
        return 0.0001;
    }
}
