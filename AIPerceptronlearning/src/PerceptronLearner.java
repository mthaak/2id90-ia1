
import java.util.List;

public class PerceptronLearner {

    /**
     * The method that implements perceptron learning
     *
     * @return answer string
     */
    public String execute(List<PVector> positive, List<PVector> negative, Boolean bias, Integer maxIterations, List<PVector> queries) {
        // If bias, add 1 to all PVectors
        if (bias) {
            for (PVector x : positive) {
                x.addCoord(1);
            }
            for (PVector x : negative) {
                x.addCoord(1);
            }
            for (PVector x : queries) {
                x.addCoord(1);
            }
        }

        // Initiate w
        int n = positive.get(0).size();
        PVector w = PVector.constant(n, 1);

        // Learn w
        boolean wChanged = false; // whether w was changed last iteration
        int iterations = 0;
        do {
            wChanged = false;
            for (PVector x : positive) {
                if (w.dotProduct(x) <= 0) {
                    w = w.add(x);
                    wChanged = true;
                }
            }
            for (PVector x : negative) {
                if (w.dotProduct(x) > 0) {
                    w = w.subtract(x);
                    wChanged = true;
                }
            }

            iterations++;
        } while (wChanged && iterations < maxIterations);

        // Initiate stringbuilder
        StringBuilder sb = new StringBuilder();
        sb.append(iterations);
        if (iterations == maxIterations) {
            return sb.toString();
        }

        sb.append(" ");
        // Classify queries
        for (PVector x : queries) {
            if (w.dotProduct(x) > 0) {
                sb.append("+");
            } else {
                sb.append("-");
            }
        }

        return sb.toString();
    }
}
