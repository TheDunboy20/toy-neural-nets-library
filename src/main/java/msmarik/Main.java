package msmarik;

import msmarik.activations.Activations;
import msmarik.losses.Losses;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        double[] input = {1, 1, 1, 1};
        double[] label = new double[1];

        final Layer layer1 = new Layer(10, Activations.relu());
        final Layer layer2 = new Layer(20, Activations.relu());
        final Layer layer3 = new Layer(1, Activations.linear());

        final Net net = new Net(Losses.MSE(), 0.001);
        net.addLayer(layer1);
        net.addLayer(layer2);
        net.addLayer(layer3);

        double[] result = net.forward(input);
        double loss = net.calculateLoss(result, label);


        System.out.println("Neural network result:" + Arrays.toString(result));
        System.out.println("Calculated loss: " + loss);

    }
}