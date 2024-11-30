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

[TestFixture]
public class AnnTests
{
    private static readonly VectorBuilder<double> _v = Vector<double>.Build;
    private static readonly MatrixBuilder<double> _m = Matrix<double>.Build;
    private static readonly Matrix<double> _layerMatrix = _m.DenseOfArray(
        new double[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 },
        }
    );

    private static readonly double[] _inputsArray = [11, 22, 33];
    private readonly List<Layer> _layers =
    [
        Layer.Create(_layerMatrix, ActivationFunction),
        Layer.Create(_layerMatrix.Multiply(2), ActivationFunction),
    ];
    private readonly Vector<double> _inputs = _v.Dense(_inputsArray);
    private readonly ITrainer _trainer = new BackPropagationWithGradientDescent(
        new FlatLearningRate(0.9),
        new MeanSquaredError(),
        new SupervisedLearningData(
            Matrix<double>.Build.Random(0, 0),
            Matrix<double>.Build.Random(0, 0),
            100,
            0.01
        ),
        global::Learning.Supervised.Ann.Ann.Create()
    );

    private static readonly IActivationFunction _activator = new LinearActivator();
    private const ActivationFunction ActivationFunction = IActivationFunction
        .ActivationFunction
        .Linear;

    [Test]
    public void Create_Should_SuccessfullyCreateAnn()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        ann.Layers.Should().BeEmpty();
    }

    [Test]
    public void CreateWithParams_Should_SuccessfullyCreateAnn()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create(_layers, _trainer);
        ann.Layers.Should().BeEquivalentTo(_layers);
    }

    [Test]
    public void AddLayer_Should_AddLayer()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        foreach (var layer in _layers)
            ann.AddLayer(layer);

        ann.Layers.Should().BeEquivalentTo(_layers);
    }

    [Test]
    public void AddLayers_Should_AddLayers()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create().AddLayers(_layers);

        ann.Layers.Should().BeEquivalentTo(_layers);
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
    public void Build_Should_CreateGraph_When_LayersAreAlreadyBuilt()
    {
        const ActivationFunction activatorFinal = ActivationFunction.Sigmoid;

        const int inputsSize = 2;
        var inputs = _v.DenseOfArray([0.1, 0.9]);

        const int firstLayerSize = 3;
        const int secondLayerSize = 4;

        var firstLayer = Layer
            .CreateWithRandomWeights(firstLayerSize, inputsSize, 0, 1, ActivationFunction)
            .BuildWeights()
            .SetInputs(inputs);

        var secondLayer = Layer
            .CreateWithRandomWeights(secondLayerSize, firstLayerSize, 0, 1, activatorFinal)
            .BuildWeights()
            .AddParentLayer(firstLayer);

        var layers = new List<Layer> { firstLayer.Clone(), secondLayer.Clone() };
        var ann = global::Learning.Supervised.Ann.Ann.Create(layers, _trainer).Build();
        ann.Layers.First().SetInputs(inputs);

        ann.Layers.Should().HaveCount(2);
        ann.Layers.Should().BeEquivalentTo(new List<Layer> { firstLayer, secondLayer });
    }

    [Test]
    public void Build_Should_CreateGraph_When_LayersAreNotBuilt()
    {
        const ActivationFunction activatorFinal = ActivationFunction.Sigmoid;

        const int inputsSize = 2;
        var inputs = _v.DenseOfArray([0.1, 0.9]);

        const int firstLayerSize = 3;
        const int secondLayerSize = 4;
        var firstLayer = Layer.CreateWithRandomWeights(
            firstLayerSize,
            inputsSize,
            0,
            1,
            ActivationFunction
        );
        var secondLayer = Layer.CreateWithRandomWeights(
            secondLayerSize,
            firstLayerSize,
            0,
            1,
            activatorFinal
        );

        var layers = new List<Layer> { firstLayer.Clone(), secondLayer.Clone() };
        var ann = global::Learning.Supervised.Ann.Ann.Create(layers, _trainer).Build();
        ann.Run(inputs);

        ann.Layers.Should().HaveCount(2);
        ann.Layers.Should()
            .BeEquivalentTo(
                new List<Layer>
                {
                    firstLayer.BuildWeights().SetInputs(inputs),
                    secondLayer.BuildWeights().AddParentLayer(firstLayer),
                }
            );
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
        var activatorFinal = ActivationFunction.Sigmoid;

        var inputsSize = 2;
        var inputs = _v.DenseOfArray([0.1, 0.9]);

        var firstLayerSize = 3;
        var secondLayerSize = 4;
        var firstLayer = Layer.CreateWithRandomWeights(
            firstLayerSize,
            inputsSize,
            0,
            1,
            ActivationFunction
        );
        var secondLayer = Layer.CreateWithRandomWeights(
            secondLayerSize,
            firstLayerSize,
            0,
            1,
            activatorFinal
        );

        var layers = new List<Layer> { firstLayer.Clone(), secondLayer.Clone() };

        var ann = global::Learning.Supervised.Ann.Ann.Create(layers, _trainer).Build();
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
        var inputs = _v.DenseOfArray([0.5, 1.0]);

        var firstLayerWeights = new[,]
        {
            { 0, 0.5, 1.0 },
            { 1.0, 0.5, 0 },
        };
        var secondLayerWeights = new[,]
        {
            { 0, 0.5, 1.0 },
        };

        var firstLayer = Layer.Create(_m.DenseOfArray(firstLayerWeights), ActivationFunction);
        var secondLayer = Layer.Create(_m.DenseOfArray(secondLayerWeights), ActivationFunction);

        var layers = new List<Layer> { firstLayer, secondLayer };
        var ann = global::Learning.Supervised.Ann.Ann.Create(layers, _trainer).Build();
        ann.Run(inputs);

        var result = ann.Outputs;

        result.Should().BeEquivalentTo(_v.DenseOfArray([1.875]));
    }

    [Test]
    public void Run_Should_Fail_When_NoInputs()
    {
        var act = () =>
            global::Learning
                .Supervised.Ann.Ann.Create(_layers, _trainer)
                .Build()
                .Run(Vector<double>.Build.Random(0));

        act.Should()
            .Throw<InvalidOperationException>()
            .WithMessage("Learning.Supervised.Ann must have inputs to run");
    }
}
