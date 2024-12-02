using Learning.Supervised.Ann.Structure;
using Learning.Supervised.Training.Algorithm.Interface;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Supervised.Ann;

public class Ann
{
    private Vector<double> _outputs = null!;
    private ITrainer _trainer = null!;

    private Ann() { }

    private Ann(List<Layer> layers, ITrainer trainer)
    {
        AddLayers(layers);
        SetTrainer(trainer);
    }

    public List<Layer> Layers { get; } = [];
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
    public void Run(Vector<double> inputs)
    {
        if (!HasBeenBuilt)
            throw new InvalidOperationException(
                "Learning.Supervised.Ann must be built before being run"
            );
        if (inputs.Count == 0)
            throw new InvalidOperationException("Learning.Supervised.Ann must have inputs to run");

        // Run first layer with inputs
        Layers.First().Activate(inputs);
        for (var layer = 1; layer < Layers.Count; layer++)
        {
            Layers[layer].Activate();
        }

        _outputs =
            Layers.Last().Outputs
            ?? throw new InvalidOperationException("Failed to retrieve final layer output vector");

        HasRun = true;
    }

    public static Ann Create()
    {
        return new Ann();
    }

    public static Ann Create(List<Layer> layers, ITrainer trainer)
    {
        return new Ann(layers, trainer);
    }

    public Ann AddLayer(Layer layer)
    {
        HasBeenBuilt = false;
        HasRun = false;
        Layers.Add(layer);
        return this;
    }

    public Ann AddLayers(List<Layer> layers)
    {
        foreach (var layer in layers)
            AddLayer(layer);
        return this;
    }

    public Ann SetTrainer(ITrainer trainer)
    {
        _trainer = trainer;
        return this;
    }

    /// <summary>
    ///     Build the Learning.Supervised.Ann graph from the weights and inputs
    /// </summary>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public Ann Build()
    {
        if (HasBeenBuilt)
            return this;

        if (Layers.Count == 0)
            throw new InvalidOperationException(
                "Learning.Supervised.Ann must have layers to build"
            );

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
    ///     Trains the Learning.Supervised.Ann using the trainer
    /// </summary>
    public void Train()
    {
        _trainer.Train();
    }
}
