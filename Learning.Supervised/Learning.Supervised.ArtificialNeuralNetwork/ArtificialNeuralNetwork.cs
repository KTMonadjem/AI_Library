using System.Text;
using Learning.Supervised.ArtificialNeuralNetwork.Structure;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.ArtificialNeuralNetwork;

public class Ann
{
    private Vector<double> _outputs = null!;

    private Ann() { }

    private Ann(List<Layer> layers)
    {
        AddLayers(layers);
    }

    /// <summary>
    /// The list of layers in this ANN
    /// </summary>
    public List<Layer> Layers { get; } = [];

    /// <summary>
    /// A flag indicating whether this ANN has been built or not
    /// Is required to be built to run
    /// Has to be rebuilt if new layers are added
    /// </summary>
    public bool HasBeenBuilt { get; private set; }

    /// <summary>
    /// The outputs from the last run of this ANN
    /// </summary>
    /// <exception cref="InvalidOperationException"></exception>
    public Vector<double> Outputs
    {
        get
        {
            if (_outputs is null)
                throw new InvalidOperationException(
                    "ANN must be run before before outputs can be read"
                );
            return _outputs;
        }
    }

    /// <summary>
    ///     Run the inputs through this Learning.Supervised.ArtificialNeuralNetwork into the output
    /// </summary>
    /// <exception cref="InvalidOperationException"></exception>
    public void Run(Vector<double> inputs)
    {
        if (!HasBeenBuilt)
            throw new InvalidOperationException("ANN must be built before being run");
        if (inputs.Count <= 0)
            throw new InvalidOperationException("ANN must have more than 0 inputs.");

        // Run first layer with inputs
        Layers.First().Activate(inputs);
        for (var layer = 1; layer < Layers.Count; layer++)
        {
            Layers[layer].Activate();
        }

        _outputs =
            Layers.Last().Outputs
            ?? throw new InvalidOperationException("Failed to retrieve final layer output vector");
    }

    /// <summary>
    /// Creates a new, empty ANN
    /// </summary>
    /// <returns></returns>
    public static Ann Create()
    {
        return new Ann();
    }

    /// <summary>
    /// Creates an ANN from the specified layers
    /// </summary>
    /// <param name="layers"></param>
    /// <returns></returns>
    public static Ann Create(List<Layer> layers)
    {
        return new Ann(layers);
    }

    /// <summary>
    /// Adds a new layer to the ANN
    /// </summary>
    /// <param name="layer"></param>
    /// <returns></returns>
    public Ann AddLayer(Layer layer)
    {
        HasBeenBuilt = false;
        Layers.Add(layer);
        return this;
    }

    /// <summary>
    /// Adds a range of layers to the ANN
    /// </summary>
    /// <param name="layers"></param>
    /// <returns></returns>
    public Ann AddLayers(List<Layer> layers)
    {
        foreach (var layer in layers)
            AddLayer(layer);
        return this;
    }

    /// <summary>
    /// Build the ANN graph from the weights and inputs
    /// </summary>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public Ann Build()
    {
        if (HasBeenBuilt)
            return this;

        if (Layers.Count == 0)
            throw new InvalidOperationException("ANN must have layers to build");

        Layer? previous = null;
        foreach (var layer in Layers)
        {
            if (previous is not null)
            {
                previous.SetOutputLayer(layer);
                layer.SetInputLayer(previous);
            }

            previous = layer;
        }

        HasBeenBuilt = true;
        return this;
    }

    /// <summary>
    /// Outputs the layer weights for each layer in this ANN
    /// <seealso cref="Layer.ToString"/>
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
        var sb = new StringBuilder();
        foreach (var layer in Layers)
        {
            sb.Append(layer);
            sb.AppendLine("|".PadLeft(10));
            sb.AppendLine("|".PadLeft(10));
        }

        return sb.ToString();
    }
}
