package msmarik;

import msmarik.initializers.Initializer;
import msmarik.losses.Loss;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class Net {
    private final ArrayList<Layer> layers;
    private final Loss lossFn;
    private final Initializer weightInitializer;

    Net(Builder builder) {
        this.layers = new ArrayList<>(builder.layers);
        this.lossFn = builder.lossFn;
        this.weightInitializer = builder.weightInitializer;
        validateNetStructure();
        initializeNet();
    }

    public static class Builder {
        private final ArrayList<Layer> layers = new ArrayList<>();
        private Loss lossFn;
        private Initializer weightInitializer;

        public Builder addLayer(Layer layer) {
            layers.add(Objects.requireNonNull(layer, "Layer cannot be null"));
            return this;
        }

        public Builder lossFn(Loss lossFn) {
            this.lossFn = lossFn;
            return this;
        }

        public Builder weightInitializer(Initializer weightInitializer) {
            this.weightInitializer = weightInitializer;
            return this;
        }

        public Net build() {
            return new Net(this);
        }
    }

    public double trainStep(double[] input, double[] correctLabels, double learningRate) {
        validateLearningRate(learningRate);
        validateInputAndOutputShape(input, correctLabels);

        double[] predictedProbabilities = forward(input);
        double preUpdateLoss = calculateLoss(predictedProbabilities, correctLabels);
        backpropagate(correctLabels);
        updateWeightsAndBiases(learningRate);

        return preUpdateLoss;
    }

    @SuppressWarnings({"unchecked"})
    public List<Layer> getLayers() {
        return (List<Layer>) this.layers.clone();
    }

    public Layer getLayer(int index) {
        return this.layers.get(index);
    }

    public double[] forward(double[] input) {
        double[] result = input.clone();

        for (Layer layer : layers) {
            result = layer.forward(result);
        }

        return result;
    }

    public double calculateLoss(double[] predictedProbabilities, double[] correctLabels) {
        return this.lossFn.standard(predictedProbabilities, correctLabels);
    }

    public void backpropagate(double[] correctLabels) {
        Layer outputLayer = layers.getLast();
        outputLayer.backpropagateOutputLayer(correctLabels, lossFn);

        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer current = layers.get(i);
            Layer next = layers.get(i + 1);

            current.backpropagateHiddenLayer(next);
        }
    }

    public void updateWeightsAndBiases(double learningRate) {
        for (Layer layer : layers) {
            layer.updateWeightsAndBiases(learningRate);
        }
    }

    private void initializeNet() {
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            layer.initializeLayer(weightInitializer);
            layer.setLayerName("Layer-" + i); // TODO: Refactor this to pass layerIndex inside
        }
    }

    private void validateNetStructure() {
        if (this.lossFn == null) throw new IllegalStateException("Loss function must be provided");
        if (this.layers.isEmpty()) throw new IllegalStateException("At least one layer must exist");
    }

    private void validateInputAndOutputShape(double[] input, double[] correctLabels) {
        if (input == null) throw new IllegalArgumentException("Input to the network cannot be null");
        if (correctLabels == null) throw new IllegalArgumentException("The correct labels can not be null");

        final Layer firstLayer = layers.getFirst();

        if (input.length != firstLayer.getWeightsNumber()) {
            throw new IllegalArgumentException("Input shape: [" + input.length + "] must match the first layer shape: [" + firstLayer.getWeightsNumber() + "]");
        }

        final Layer outputLayer = layers.getLast();
        if (correctLabels.length != outputLayer.getPerceptronsNumber()) {
            throw new IllegalArgumentException("Correct labels shape: [" + correctLabels.length + "] must match the output layer shape: [" + outputLayer.getPerceptronsNumber() + "]");
        }
    }

    private void validateLearningRate(double learningRate) {
        if (!Double.isFinite(learningRate) || learningRate <= 0) throw new IllegalArgumentException("Learning rate must be finite and greater than 0.");
    }
}
