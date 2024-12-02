using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Ann.Structure;

// Weights are row-wise per neuron, column-wise per input. Last column is the bias.
using Weights = Matrix<double>;

public class Layer
{
    private static readonly Random Random = new Random();
    private static readonly VectorBuilder<double> VectorBuilder = Vector<double>.Build;

    public Layer? InputLayer { get; private set; }
    public Layer? OutputLayer { get; private set; }

    public Weights InputWeights { get; private set; }
    public Weights? OutputWeights => OutputLayer?.InputWeights;

    public IActivationFunction ActivationFunction { get; private set; }

    public Vector<double>? Deltas { get; private set; }
    public Vector<double>? Gradients { get; private set; }

    public Vector<double>? Inputs { get; private set; }
    public Vector<double>? Outputs { get; private set; }
    public Vector<double>? Derivatives { get; private set; }

    private Layer(Weights inputWeights, IActivationFunction activationFunction)
    {
        InputWeights = inputWeights;
        ActivationFunction = activationFunction;
    }

    public static Layer Create(Weights inputWeights, IActivationFunction activationFunction)
    {
        return new Layer(inputWeights, activationFunction);
    }

    /// <summary>
    ///     Creates a layer with randomized weights. Additionally creates bias weights.
    /// </summary>
    /// <param name="numberOfNeurons"></param>
    /// <param name="numberOfWeights">The number of weights minus the bias weight</param>
    /// <param name="minWeight"></param>
    /// <param name="maxWeight"></param>
    /// <param name="activationFunction"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static Layer CreateWithRandomWeights(
        int numberOfNeurons,
        int numberOfWeights,
        double minWeight,
        double maxWeight,
        IActivationFunction activationFunction
    )
    {
        if (numberOfNeurons <= 0)
            throw new ArgumentException("Layer must be created with neurons");
        if (numberOfWeights <= 0)
            throw new ArgumentException("Layer must be created with weights");

        if (minWeight > maxWeight)
            throw new ArgumentException("Min weight must be less than max weight");

        // Add the bias neuron
        numberOfWeights++;

        var weights = new double[numberOfNeurons, numberOfWeights];
        for (var i = 0; i < numberOfNeurons; i++)
        for (var j = 0; j < numberOfWeights; j++)
            // Assign a randomly generated weight
            weights[i, j] = Random.NextDouble() * (maxWeight - minWeight) + minWeight;

        return Create(Matrix<double>.Build.DenseOfArray(weights), activationFunction);
    }

    public Layer SetInputLayer(Layer layer)
    {
        InputLayer = layer;
        return this;
    }

    public Layer SetOutputLayer(Layer layer)
    {
        OutputLayer = layer;
        return this;
    }

    public void Activate(Vector<double>? inputs = null)
    {
        Inputs = inputs ?? InputLayer?.Outputs ?? throw new ArgumentNullException(nameof(inputs));

        var inputsWithBias = Inputs.Add(1);

        var summedInputs = InputWeights.Multiply(inputsWithBias);

        // TODO: Vector activation operations
        var activations = new double[summedInputs.Count];
        var derivatives = new double[summedInputs.Count];
        for (var neuron = 0; neuron < summedInputs.Count; neuron++)
        {
            (activations[neuron], derivatives[neuron]) = ActivationFunction.Activate(
                summedInputs[neuron]
            );
        }

        Outputs = VectorBuilder.DenseOfArray(activations);
        Derivatives = VectorBuilder.DenseOfArray(derivatives);
    }
}
