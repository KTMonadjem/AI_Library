using System.Text;
using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.ArtificialNeuralNetwork.Structure;

using Weights = Matrix<double>;

public class Layer
{
    private static readonly Random Random = new();
    private static readonly VectorBuilder<double> VectorBuilder = Vector<double>.Build;

    /// <summary>
    /// The length of this layer in neurons
    /// </summary>
    public int NumberOfNeurons { get; private set; }

    /// <summary>
    /// The reference to the layer before this, if it exists
    /// </summary>
    public Layer? InputLayer { get; private set; }

    /// <summary>
    /// The reference to the layer after this, if it exists
    /// </summary>
    public Layer? OutputLayer { get; private set; }

    /// <summary>
    /// The weights leading to this layer
    /// Weights are row-wise per neuron, column-wise per input. Last column is the bias.
    /// </summary>
    public Weights InputWeights { get; set; }

    /// <summary>
    /// The weights leading to the following layer
    /// Weights are row-wise per neuron, column-wise per input. Last column is the bias.
    /// </summary>
    public Weights? OutputWeights => OutputLayer?.InputWeights;

    /// <summary>
    /// The activation function for each neuron in this layer
    /// </summary>
    public IActivationFunction ActivationFunction { get; }

    /// <summary>
    /// The weight update (delta) for each input weight to this layer
    /// </summary>
    public Matrix<double>? Deltas { get; set; }

    /// <summary>
    /// The error gradient for each neuron on this layer
    /// </summary>
    public Vector<double>? Gradients { get; set; }

    /// <summary>
    /// The input vector to this layer for each neuron's previous activation
    /// <see cref="Activate"/>
    /// </summary>
    public Vector<double>? Inputs { get; private set; }

    /// <summary>
    /// The output vector from this layer for each neuron's previous activation
    /// <see cref="Activate"/>
    /// </summary>
    public Vector<double>? Outputs { get; private set; }

    /// <summary>
    /// The vector of derivatives for each neuron's previous activation
    /// <see cref="Activate"/>
    /// </summary>
    public Vector<double>? Derivatives { get; private set; }

    private Layer(Weights inputWeights, IActivationFunction activationFunction)
    {
        InputWeights = inputWeights;
        NumberOfNeurons = inputWeights.RowCount;
        ActivationFunction = activationFunction;
    }

    /// <summary>
    /// Initialise a new layer with the given input weights and activation function
    /// </summary>
    /// <param name="inputWeights">A matrix of input weights</param>
    /// <param name="activationFunction">The activation function for each neuron</param>
    /// <returns>A new layer that has not been activated</returns>
    public static Layer Create(Weights inputWeights, IActivationFunction activationFunction)
    {
        return new Layer(inputWeights, activationFunction);
    }

    /// <summary>
    /// Creates a layer with randomized weights.
    /// Creates bias weights.
    /// </summary>
    /// <param name="numberOfNeurons">The number of neurons in this layer</param>
    /// <param name="numberOfInputs">The number of input weights to this layer</param>
    /// <param name="activationFunction">The activation function for each neuron</param>
    /// <param name="maxWeight">The maximum value for any weight. Default is 1.0</param>
    /// <param name="minWeight">The minimum value for any weight. Default is 0.0</param>
    /// <returns>A new layer with random weights that has not been activated</returns>
    /// <exception cref="ArgumentException">Number of inputs is invalid</exception>
    /// <exception cref="ArgumentException">Number of neurons is invalid</exception>
    /// <exception cref="ArgumentException">Minimum weight is greater than or equal to maximum</exception>
    public static Layer CreateWithRandomWeights(
        int numberOfNeurons,
        int numberOfInputs,
        IActivationFunction activationFunction,
        double maxWeight = 1,
        double minWeight = 0
    )
    {
        if (numberOfNeurons <= 0)
            throw new ArgumentException("Layer must be created with neurons");

        if (numberOfInputs <= 0)
            throw new ArgumentException("Layer must be created with inputs");

        if (minWeight >= maxWeight)
            throw new ArgumentException("Minimum weight must be greater than maximum weight");

        var layerWeights = Weights.Build.Dense(
            numberOfNeurons, // Add a bias neuron weight
            numberOfInputs + 1,
            (_, _) => Random.NextDouble() * (maxWeight - minWeight) + minWeight
        );

        return new Layer(layerWeights, activationFunction);
    }

    /// <summary>
    /// Sets the input layer reference for this layer
    /// </summary>
    /// <param name="layer">The layer that points to this layer</param>
    /// <returns></returns>
    public Layer SetInputLayer(Layer layer)
    {
        InputLayer = layer;
        return this;
    }

    /// <summary>
    /// Sets the output layer reference for this layer
    /// </summary>
    /// <param name="layer">The layer that this layer points to</param>
    /// <returns></returns>
    public Layer SetOutputLayer(Layer layer)
    {
        OutputLayer = layer;
        return this;
    }

    /// <summary>
    /// Activate the layer with either the inputs given,
    /// or using the previous layer's outputs
    /// </summary>
    /// <param name="inputs">The inputs to use, or null if using the previous layer's outputs</param>
    /// <exception cref="ArgumentNullException">No inputs have been provided and the previous layer does not have outputs</exception>
    public void Activate(Vector<double>? inputs = null)
    {
        Inputs = inputs ?? InputLayer?.Outputs ?? throw new ArgumentNullException(nameof(inputs));

        // Add a bias input
        Inputs = Vector<double>.Build.DenseOfEnumerable(Inputs.Append(1.0));

        var summedInputs = InputWeights.Multiply(Inputs);

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

    /// <summary>
    /// Returns a string of the layer weights in the following format:
    ///     [0.5, 0.44, -0.1], -> Weights to neuron 1
    ///     [0.3, 0.4, 0.66] -> Weights to neuron 2
    /// </summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        foreach (var row in InputWeights.EnumerateRows())
        {
            sb.AppendLine($"[{string.Join(",", row)}],");
        }

        return sb.ToString();
    }
}
