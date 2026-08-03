package msmarik.demo;

import msmarik.Layer;
import msmarik.Net;
import msmarik.activations.Activations;
import msmarik.initializers.Initializers;
import msmarik.losses.Losses;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        double[] input = {1, 1, 1, 1};
        double[] label = new double[1];

        Net net  = new Net.Builder()
                .weightInitializer(Initializers.he())
                .addLayer(new Layer(4, 10, Activations.relu()))
                .addLayer(new Layer(10, 20, Activations.relu()))
                .addLayer(new Layer(20, 1, Activations.linear()))
                .lossFn(Losses.MSE())
                .build();

        double[] result = net.forward(input);
        double loss = net.calculateLoss(result, label);


        System.out.println("Neural network result:" + Arrays.toString(result));
        System.out.println("Calculated loss: " + loss);

    }
}
