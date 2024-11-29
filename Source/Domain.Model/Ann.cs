using Learning.Supervised.Ann.Interface;
using Learning.Supervised.Ann.Structure;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Ann;

public class Ann : IAnn
{
    private bool _inputsModified = true;
    private Vector<double> _outputs = null!;

    private Ann() { }

    private Ann(List<Layer> layers, Vector<double> inputs)
    {
        SetInputs(inputs);
        AddLayers(layers);
    }

    public List<Layer> Layers { get; } = new();
    public Vector<double> Inputs { get; private set; } = Vector<double>.Build.Dense([]);
    public bool HasRun { get; private set; }
    public bool HasBeenBuilt { get; private set; }

    public Vector<double> Outputs
    {
        get
        {
            if (!HasRun)
                throw new InvalidOperationException(
                    "Learning.Supervised.Ann must be run before before outputs can be read"
                );
            return _outputs;
        }
    }

    /// <summary>
    ///     Run the inputs through this Learning.Supervised.Ann into the output
    /// </summary>
    /// <exception cref="InvalidOperationException"></exception>
    public void Run()
    {
        if (!HasBeenBuilt)
            throw new InvalidOperationException(
                "Learning.Supervised.Ann must be built before being run"
            );

        // We only care about the last layers neurons.
        var finalLayerCount = Layers.Last().Neurons.Count;
        var outputs = new double[Layers.Last().Neurons.Count];
        for (var i = 0; i < finalLayerCount; i++)
            // Fetching a neuron's output will fetch the parent's output too
            outputs[i] = Layers.Last().Neurons[i].Output;

        _outputs = Vector<double>.Build.Dense(outputs);

        HasRun = true;
        _inputsModified = false;
    }

    public static Ann Create()
    {
        return new Ann();
    }

    public static Ann Create(List<Layer> layers, Vector<double> inputs)
    {
        return new Ann(layers, inputs);
    }

    public Ann AddLayer(Layer layer)
    {
        HasBeenBuilt = false;
        _inputsModified = true;
        Layers.Add(layer);
        return this;
    }

    public Ann AddLayers(List<Layer> layers)
    {
        foreach (var layer in layers)
            AddLayer(layer);
        return this;
    }

    public Ann SetInputs(Vector<double> inputs)
    {
        Inputs = inputs;
        return this;
    }

    /// <summary>
    ///     Build the Learning.Supervised.Ann graph from the weights and inputs
    /// </summary>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public Ann Build()
    {
        if (!Layers.Any())
            throw new InvalidOperationException(
                "Learning.Supervised.Ann must have layers to build"
            );
        if (!Inputs.Any())
            throw new InvalidOperationException(
                "Learning.Supervised.Ann must have inputs to build"
            );

        Layer? previous = null;
        foreach (var layer in Layers)
        {
            if (!layer.IsBuilt)
                layer.BuildWeights();
            if (!layer.HasInputs)
            {
                if (previous is null)
                    // If this is the first layer, use inputs instead of parents
                    layer.AddInputs(Inputs);
                else
                    // Otherwise add parents to the current layer's inputs
                    layer.AddParentLayer(previous);
            }

            previous = layer;
        }

        HasBeenBuilt = true;
        return this;
    }

    /// <summary>
    ///     Trains the Learning.Supervised.Ann using the trainer
    /// </summary>
    public void Train() { }
}
