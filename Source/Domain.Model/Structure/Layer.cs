using Common.Maths.ActivationFunction.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Ann.Structure;

// Weights are row-wise per neuron, column-wise per input. Last column is the bias.
using Weights = Matrix<double>;

public class Layer
{
    public int NumberOfNeurons { get; private set; }

    private static readonly Random Random = new Random();
    private static readonly VectorBuilder<double> VectorBuilder = Vector<double>.Build;

    public Layer? InputLayer { get; private set; }
    public Layer? OutputLayer { get; private set; }

    public Weights? InputWeights { get; set; }
    public Weights? OutputWeights => OutputLayer?.InputWeights;

    public IActivationFunction ActivationFunction { get; private set; }

    public Matrix<double>? Deltas { get; set; }
    public Vector<double>? Gradients { get; set; }

    public Vector<double>? Inputs { get; private set; }
    public Vector<double>? Outputs { get; private set; }
    public Vector<double>? Derivatives { get; private set; }

    private Layer(Weights inputWeights, IActivationFunction activationFunction)
    {
        InputWeights = inputWeights;
        NumberOfNeurons = inputWeights.RowCount;
        ActivationFunction = activationFunction;
    }

    private Layer(int numberOfNeurons, IActivationFunction activationFunction)
    {
        NumberOfNeurons = numberOfNeurons;
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
    /// <param name="activationFunction"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static Layer CreateWithRandomWeights(
        int numberOfNeurons,
        IActivationFunction activationFunction
    )
    {
        if (numberOfNeurons <= 0)
            throw new ArgumentException("Layer must be created with neurons");

        return new Layer(numberOfNeurons, activationFunction);
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

        Inputs = Vector<double>.Build.DenseOfEnumerable(Inputs.Append(1.0));

        var summedInputs = InputWeights!.Multiply(Inputs);

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
