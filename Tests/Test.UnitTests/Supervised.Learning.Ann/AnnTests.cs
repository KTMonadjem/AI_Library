using Common.Maths.ActivationFunction;
using Common.Maths.ActivationFunction.Interface;
using FluentAssertions;
using Learning.Supervised.Ann.Structure;
using MathNet.Numerics.LinearAlgebra;
using static Common.Maths.ActivationFunction.Interface.IActivationFunction;

namespace Tests.Supervised.Learning.Ann;

[TestFixture]
public class AnnTests
{
    [SetUp]
    public void SetUp()
    {
        _layers = new List<Layer>
        {
            Layer.Create(_layerMatrix, _activationFunction),
            Layer.Create(_layerMatrix.Multiply(2), _activationFunction),
        };
        _inputs = V.Dense(_inputsArray);
    }

    private static readonly VectorBuilder<double> V = Vector<double>.Build;
    private static readonly MatrixBuilder<double> M = Matrix<double>.Build;

    private static readonly Matrix<double> _layerMatrix = M.DenseOfArray(
        new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 },
        }
    );

    private static readonly double[] _inputsArray = { 11, 22, 33 };

    private List<Layer> _layers;
    private Vector<double> _inputs;
    private static readonly IActivationFunction _activator = new LinearActivator();
    private static readonly ActivationFunction _activationFunction = ActivationFunction.Linear;

    [Test]
    public void Create_Should_SuccessfullyCreateAnn()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        ann.Layers.Should().BeEmpty();
    }

    [Test]
    public void CreateWithParams_Should_SuccessfullyCreateAnn()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create(_layers, _inputs);
        ann.Layers.Should().BeEquivalentTo(_layers);
        ann.Inputs.Should().BeEquivalentTo(_inputs);
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
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        ann.AddLayers(_layers);

        ann.Layers.Should().BeEquivalentTo(_layers);
    }

    [Test]
    public void SetInputs_Should_SetInputs()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        ann.SetInputs(_inputs);

        ann.Inputs.Should().BeEquivalentTo(_inputs);
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
    public void Build_Should_Fail_When_NoInputs()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        ann.AddLayers(_layers);
        Action act = () => ann.Build();

        act.Should()
            .Throw<InvalidOperationException>()
            .WithMessage("Learning.Supervised.Ann must have inputs to build");
    }

    [Test]
    public void Build_Should_CreateGraph_When_LayersAreAlreadyBuilt()
    {
        var activatorFinal = ActivationFunction.Sigmoid;

        var inputsSize = 2;
        var inputs = V.DenseOfArray([0.1, 0.9]);

        var firstLayerSize = 3;
        var secondLayerSize = 4;

        var firstLayer = Layer
            .CreateWithRandomWeights(firstLayerSize, inputsSize, 0, 1, _activationFunction)
            .BuildWeights()
            .AddInputs(inputs);

        var secondLayer = Layer
            .CreateWithRandomWeights(secondLayerSize, firstLayerSize, 0, 1, activatorFinal)
            .BuildWeights()
            .AddParentLayer(firstLayer);

        var layers = new List<Layer> { firstLayer.Clone(), secondLayer.Clone() };
        var Ann = global::Learning.Supervised.Ann.Ann.Create(layers, inputs).Build();

        Ann.Inputs.Should().BeEquivalentTo(inputs);
        Ann.Layers.Should().HaveCount(2);
        Ann.Layers.Should().BeEquivalentTo(new List<Layer> { firstLayer, secondLayer });
    }

    [Test]
    public void Build_Should_CreateGraph_When_LayersAreNotBuilt()
    {
        var activatorFinal = ActivationFunction.Sigmoid;

        var inputsSize = 2;
        var inputs = V.DenseOfArray([0.1, 0.9]);

        var firstLayerSize = 3;
        var secondLayerSize = 4;
        var firstLayer = Layer.CreateWithRandomWeights(
            firstLayerSize,
            inputsSize,
            0,
            1,
            _activationFunction
        );
        var secondLayer = Layer.CreateWithRandomWeights(
            secondLayerSize,
            firstLayerSize,
            0,
            1,
            activatorFinal
        );

        var layers = new List<Layer> { firstLayer.Clone(), secondLayer.Clone() };
        var ann = global::Learning.Supervised.Ann.Ann.Create(layers, inputs).Build();
        ann.Run();

        ann.Inputs.Should().BeEquivalentTo(inputs);
        ann.Layers.Should().HaveCount(2);
        ann.Layers.Should()
            .BeEquivalentTo(
                new List<Layer>
                {
                    firstLayer.BuildWeights().AddInputs(inputs),
                    secondLayer.BuildWeights().AddParentLayer(firstLayer),
                }
            );
    }

    [Test]
    public void Ann_That_HasNotBeenBuilt_Should_NotRun()
    {
        var ann = global::Learning.Supervised.Ann.Ann.Create();
        var act = () => ann.Run();

        act.Should()
            .Throw<InvalidOperationException>()
            .WithMessage("Learning.Supervised.Ann must be built before being run");
    }

    [Test]
    public void Ann_That_HasBeenBuilt_Should_Run()
    {
        var activatorFinal = ActivationFunction.Sigmoid;

        var inputsSize = 2;
        var inputs = V.DenseOfArray([0.1, 0.9]);

        var firstLayerSize = 3;
        var secondLayerSize = 4;
        var firstLayer = Layer.CreateWithRandomWeights(
            firstLayerSize,
            inputsSize,
            0,
            1,
            _activationFunction
        );
        var secondLayer = Layer.CreateWithRandomWeights(
            secondLayerSize,
            firstLayerSize,
            0,
            1,
            activatorFinal
        );

        var layers = new List<Layer> { firstLayer.Clone(), secondLayer.Clone() };

        var ann = global::Learning.Supervised.Ann.Ann.Create(layers, inputs).Build();
        ann.Run();

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

        var firstLayer = Layer.Create(M.DenseOfArray(firstLayerWeights), _activationFunction);
        var secondLayer = Layer.Create(M.DenseOfArray(secondLayerWeights), _activationFunction);

        var layers = new List<Layer> { firstLayer, secondLayer };
        var ann = global::Learning.Supervised.Ann.Ann.Create(layers, inputs).Build();
        ann.Run();

        var result = ann.Outputs;

        result.Should().BeEquivalentTo(V.DenseOfArray(new[] { 1.875 }));
    }
}
