using Common.Maths.ActivationFunction;
using Common.Maths.ActivationFunction.Interface;
using FluentAssertions;
using Learning.Supervised.Ann.Algorithm;
using Learning.Supervised.Ann.Structure;
using Learning.Supervised.Training.Algorithm.Interface;
using Learning.Supervised.Training.Data;
using Learning.Supervised.Training.LearningRate;
using Learning.Supervised.Training.LossFunction;
using MathNet.Numerics.LinearAlgebra;
using static Common.Maths.ActivationFunction.Interface.IActivationFunction;

namespace Tests.Supervised.Learning.Ann;

// TODO: More tests

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
    private readonly ITrainer _trainer = new BackPropagationWithGradientDescent(
        new FlatLearningRate(0.9),
        new MeanSquaredError(),
        new SupervisedLearningData(
            Matrix<double>.Build.Random(0, 0),
            Matrix<double>.Build.Random(0, 0),
            100,
            0.01
        ),
        global::Learning.Supervised.Ann.Ann.Create(),
        4
    );

    [Test]
    public void Create_Should_SuccessfullyCreateAnn()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        ann.Layers.Should().BeEmpty();
    }

    [Test]
    public void CreateWithParams_Should_SuccessfullyCreateAnn()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create(Layers, _trainer);
        ann.Layers.Should().BeEquivalentTo(Layers);
    }

    [Test]
    public void AddLayer_Should_AddLayer()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        foreach (var layer in Layers)
            ann.AddLayer(layer);

        ann.Layers.Should().BeEquivalentTo(Layers);
    }

    [Test]
    public void AddLayers_Should_AddLayers()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create().AddLayers(Layers);

        ann.Layers.Should().BeEquivalentTo(Layers);
    }

    [Test]
    public void Build_Should_Fail_When_NoLayers()
    {
        Action act = () => global::Learning.Supervised.Ann.Ann.Create().Build();

        act.Should()
            .Throw<InvalidOperationException>()
            .WithMessage("Learning.Supervised.Ann must have layers to build");
    }

    [Test]
    public void Ann_That_HasNotBeenBuilt_Should_NotRun()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        var act = () => ann.Run(_inputs);

        act.Should()
            .Throw<InvalidOperationException>()
            .WithMessage("Learning.Supervised.Ann must be built before being run");
    }

    [Test]
    public void Ann_That_HasBeenBuilt_Should_Run()
    {
        var inputsSize = 2;
        var inputs = V.DenseOfArray([0.1, 0.9]);

        var firstLayerSize = 3;
        var secondLayerSize = 4;
        var firstLayer = Layer.CreateWithRandomWeights(firstLayerSize, ActivationFunction);
        var secondLayer = Layer.CreateWithRandomWeights(secondLayerSize, ActivationFunction);

        var layers = new List<Layer> { firstLayer, secondLayer };

        var ann = global::Learning
            .Supervised.Ann.Ann.Create(layers, _trainer)
            .SetNumberOfInputs(inputsSize)
            .Build();
        ann.Run(inputs);

        var act = () =>
        {
            _ = ann.Outputs;
        };

        act.Should().NotThrow<InvalidOperationException>();
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
        var ann = global::Learning
            .Supervised.Ann.Ann.Create(layers, _trainer)
            .SetNumberOfInputs(2)
            .Build();
        ann.Run(inputs);

        var result = ann.Outputs;

        result.Should().BeEquivalentTo(V.DenseOfArray([1.5]));
    }

    [Test]
    public void Run_Should_Fail_When_IncorrecNumberOfInputs()
    {
        var act = () =>
            global::Learning
                .Supervised.Ann.Ann.Create(Layers, _trainer)
                .SetNumberOfInputs(2)
                .Build()
                .Run(Vector<double>.Build.Random(0));

        act.Should()
            .Throw<InvalidOperationException>()
            .WithMessage(
                "Learning.Supervised.Ann must have the correct number of inputs: Expected: 2, Actual: 0"
            );
    }
}
