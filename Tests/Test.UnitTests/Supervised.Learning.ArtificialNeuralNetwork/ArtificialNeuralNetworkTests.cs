using Common.Maths.ActivationFunction;
using Common.Maths.ActivationFunction.Interface;
using FluentAssertions;
using Learning.Supervised.ArtificialNeuralNetwork;
using Learning.Supervised.ArtificialNeuralNetwork.Structure;
using MathNet.Numerics.LinearAlgebra;

namespace Tests.Supervised.Learning.ArtificialNeuralNetwork;

[TestFixture]
public class AnnTests
{
    private static readonly VectorBuilder<double> V = Vector<double>.Build;
    private static readonly MatrixBuilder<double> M = Matrix<double>.Build;
    private static readonly Matrix<double> LayerMatrix = M.DenseOfArray(
        new double[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 },
        }
    );

    private static readonly IActivationFunction ActivationFunction = new LinearActivator();

    private static readonly double[] InputsArray = [11, 22, 33];
    private static readonly List<Layer> Layers =
    [
        Layer.Create(LayerMatrix, ActivationFunction),
        Layer.Create(LayerMatrix.Multiply(2), ActivationFunction),
    ];
    private readonly Vector<double> _inputs = V.Dense(InputsArray);

    [Test]
    public void Create_Should_SuccessfullyCreateAnn()
    {
        var ann = Ann.Create();
        ann.Layers.Should().BeEmpty();
        ann.HasBeenBuilt.Should().BeFalse();
    }

    [Test]
    public void CreateWithParams_Should_SuccessfullyCreateAnn()
    {
        var ann = Ann.Create(Layers);
        ann.Layers.Should().BeEquivalentTo(Layers);
        ann.HasBeenBuilt.Should().BeFalse();
    }

    [Test]
    public void AddLayer_Should_AddLayer()
    {
        var ann = Ann.Create();
        foreach (var layer in Layers)
            ann.AddLayer(layer);

        ann.Layers.Should().BeEquivalentTo(Layers);
    }

    [Test]
    public void AddLayers_Should_AddLayers()
    {
        var ann = Ann.Create().AddLayers(Layers);

        ann.Layers.Should().BeEquivalentTo(Layers);
    }

    [Test]
    public void Build_Should_Fail_When_NoLayers()
    {
        Action act = () => Ann.Create().Build();

        act.Should()
            .Throw<InvalidOperationException>()
            .WithMessage("ANN must have layers to build");
    }

    [Test]
    public void Ann_That_HasNotBeenBuilt_Should_NotRun()
    {
        var ann = Ann.Create();
        var act = () => ann.Run(_inputs);

        act.Should()
            .Throw<InvalidOperationException>()
            .WithMessage("ANN must be built before being run");
    }

    [Test]
    public void AddingALayer_Should_UnbuildANN()
    {
        var ann = Ann.Create();
        ann.HasBeenBuilt.Should().BeFalse();

        ann.AddLayer(Layers[0]);
        ann.HasBeenBuilt.Should().BeFalse();

        ann.Build();
        ann.HasBeenBuilt.Should().BeTrue();

        ann.AddLayer(Layers[1]);
        ann.HasBeenBuilt.Should().BeFalse();
    }

    /// <summary>
    ///     0.0 -> 0.0            0.0 -> 0.0
    ///     0.5 ->  0.5 -> 0.25
    ///     1.0 ->  1.0 -> 1.0 -> 1.25 -> 0.5 -> 0.625
    ///     1.0 -> 1.0
    ///     0.5 ->  0.5 -> 0.25
    ///     1.0 ->  0.0 -> 0.0 -> 1.25 -> 1.0 -> 1.25    -> 1.875
    /// </summary>
    [Test]
    public void Ann_Runs_Correctly()
    {
        var inputs = V.DenseOfArray([0.5, 1.0]);

        var firstLayerWeights = new[,]
        {
            { 0, 0.5, 1.0 },
            { 1.0, 0.5, 0 },
        };
        var secondLayerWeights = new[,]
        {
            { 0, 0.5, 1.0 },
        };

        var firstLayer = Layer.Create(M.DenseOfArray(firstLayerWeights), ActivationFunction);
        var secondLayer = Layer.Create(M.DenseOfArray(secondLayerWeights), ActivationFunction);

        var layers = new List<Layer> { firstLayer, secondLayer };
        var ann = Ann.Create(layers).Build();
        ann.Run(inputs);

        var result = ann.Outputs;

        result.Should().BeEquivalentTo(V.DenseOfArray([1.5]));
    }

    [Test]
    public void Run_Should_Fail_When_IncorrectNumberOfInputs()
    {
        var act = () => Ann.Create(Layers).Build().Run(Vector<double>.Build.Random(0));

        act.Should()
            .Throw<InvalidOperationException>()
            .WithMessage("ANN must have more than 0 inputs.");
    }
}
