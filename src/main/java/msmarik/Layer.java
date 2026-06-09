package msmarik;

import msmarik.activations.Activation;
import msmarik.losses.Loss;

public class Layer {
    private int weightsNumber;
    private int perceptronsNumber;
    private final Activation activationFn;
    private String name;
    private Perceptron[] perceptrons;

    public Layer(int weightsNumber, int perceptronsNumber, Activation activationFn) {
        this.weightsNumber = weightsNumber;
        this.perceptronsNumber = perceptronsNumber;
        this.activationFn = activationFn;
    }

    /*
    * For testing purposes only
    * */
    public Layer(Perceptron[] perceptrons, Activation activationFn) {
        this.perceptrons = perceptrons;
        this.activationFn = activationFn;
    }

    public void initializeLayer(WeightInitializer weightInitializer) {
        this.perceptrons = new Perceptron[perceptronsNumber];
        for (int i = 0; i < perceptrons.length; i++) {
            Perceptron p = new Perceptron(weightsNumber, perceptronsNumber, activationFn, weightInitializer);
            perceptrons[i] = p;
        }
    }

    public Perceptron[] getPerceptrons() {
        return perceptrons;
    }

    public double[] forward(double[] input) {
        double[] output = new double[perceptrons.length];

        for (int i = 0; i < perceptrons.length; i++) {
            double perceptronOutput = perceptrons[i].computeWeightedSum(input);
            output[i] = perceptronOutput;
        }

        return output;
    }


    public void backpropagateOutputLayer(double[] target, Loss lossFn) {

        for (int i = 0; i < perceptrons.length; i++) {
            Perceptron currentPerceptron = perceptrons[i];
            double predictedProbability = currentPerceptron.getResult();
            double error = lossFn.derivative(predictedProbability, target[i]);

            currentPerceptron.updateErrorSignal(error);
        }
    }

    public void backpropagateHiddenLayer(Layer nextLayer) {
        for (int i = 0; i < perceptrons.length; i++) {
            Perceptron currentPerceptron = perceptrons[i];
            double errorFromPreviousLayer = 0;

            for (int j = 0; j < nextLayer.perceptrons.length; j++) {
                Perceptron nextPerceptron = nextLayer.perceptrons[j];
                errorFromPreviousLayer += nextPerceptron.getWeights()[i] * nextPerceptron.getErrorSignal();
            }
            currentPerceptron.updateErrorSignal(errorFromPreviousLayer);
        }
    }

    public void updateWeightsAndBiases(double learningRate) {
        for (Perceptron perceptron: perceptrons) {
            perceptron.updateWeights(learningRate);
            perceptron.updateBias(learningRate);
        }
    }

    public void setLayerName(String name) {
        this.name = name;

        for (int i = 0; i < perceptrons.length; i++){
            perceptrons[i].setName(name + " - " + i);
        }
    }

    public String getLayerName() {
        return name;
    }
}
