using Learning.Supervised.ANN.Structure;
using Common.Maths.ActivationFunction.Interface;
using Common.Maths.ActivationFunction;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using A = Learning.Supervised.ANN.ANN;

namespace Tests.Supervised.Learning.ANN
{
    [TestFixture]
    public class ANNTests
    {
        private static readonly VectorBuilder<double> V = Vector<double>.Build;
        private static readonly MatrixBuilder<double> M = Matrix<double>.Build;

        private static readonly Matrix<double> _layerMatrix = M.DenseOfArray(new double[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 }
            });
        private static readonly double[] _inputsArray = { 11, 22, 33 };

        private List<Layer> _layers;
        private Vector<double> _inputs;
        private static readonly IActivationFunction _activator = new LinearActivator();

        [SetUp]
        public void SetUp()
        {
            _layers = new List<Layer>
            {
                Layer.Create(_layerMatrix, _activator),
                Layer.Create(_layerMatrix.Multiply(2), _activator)
            };
            _inputs = V.Dense(_inputsArray);
        }

        [Test]
        public void Create_Should_SuccessfullyCreateANN() 
        {
            var ann = A.Create();
            ann.Layers.Should().BeEmpty();
        }

        [Test]
        public void Create_Should_SuccessfullyCreateANNWithLayersAndInputs()
        {
            var ann = A.Create(_layers, _inputs);
            ann.Layers.Should().BeEquivalentTo(_layers);
            ann.Inputs.Should().BeEquivalentTo(_inputs);
        }

        [Test]
        public void AddLayer_Should_AddLayer()
        {
            var ann = A.Create();
            foreach (var layer in _layers)
            {
                ann.AddLayer(layer);
            }

            ann.Layers.Should().BeEquivalentTo(_layers);
        }

        [Test]
        public void AddLayers_Should_AddLayers()
        {
            var ann = A.Create();
            ann.AddLayers(_layers);

            ann.Layers.Should().BeEquivalentTo(_layers);
        }

        [Test]
        public void SetInputs_Should_SetInputs()
        {
            var ann = A.Create();
            ann.SetInputs(_inputs);

            ann.Inputs.Should().BeEquivalentTo(_inputs);
        }

        [Test]
        public void Build_Should_Fail_When_NoLayers()
        {
            Action act = () => A.Create().Build();

            act.Should().Throw<InvalidOperationException>().WithMessage("ANN must have layers to build");
        }

        [Test]
        public void Build_Should_Fail_When_NoInputs()
        {
            var ann = A.Create();
            ann.AddLayers(_layers);
            Action act = () => ann.Build();

            act.Should().Throw<InvalidOperationException>().WithMessage("ANN must have inputs to build");
        }

        [Test]
        public void Build_Should_CreateGraph_When_LayersAreAlreadyBuilt()
        {
            var activatorFinal = new SigmoidActivator();

            var inputsSize = 2;
            var inputs = V.DenseOfArray(new double[] { 0.1, 0.9 });

            var firstLayerSize = 3;
            var secondLayerSize = 4;

            var firstLayer = Layer.CreateWithRandomWeights(firstLayerSize, inputsSize, 0, 1, _activator)
                .BuildWeights()
                .AddInputs(inputs);

            var secondLayer = Layer.CreateWithRandomWeights(secondLayerSize, firstLayerSize, 0, 1, activatorFinal)
                .BuildWeights()
                .AddParentLayer(firstLayer);

            var layers = new List<Layer> { firstLayer.Clone(), secondLayer.Clone() };
            var ann = A.Create(layers, inputs).Build();

            ann.Inputs.Should().BeEquivalentTo(inputs);
            ann.Layers.Should().HaveCount(2);
            ann.Layers.Should().BeEquivalentTo(new List<Layer> { firstLayer, secondLayer });
        }

        [Test]
        public void Build_Should_CreateGraph_When_LayersAreNotBuilt()
        {
            var activatorFinal = new SigmoidActivator();

            var inputsSize = 2;
            var inputs = V.DenseOfArray(new double[] { 0.1, 0.9 });

            var firstLayerSize = 3;
            var secondLayerSize = 4;
            var firstLayer = Layer.CreateWithRandomWeights(firstLayerSize, inputsSize, 0, 1, _activator);
            var secondLayer = Layer.CreateWithRandomWeights(secondLayerSize, firstLayerSize, 0, 1, activatorFinal);

            var layers = new List<Layer> { firstLayer.Clone(), secondLayer.Clone() };
            var ann = A.Create(layers, inputs).Build();

            ann.Inputs.Should().BeEquivalentTo(inputs);
            ann.Layers.Should().HaveCount(2);
            ann.Layers.Should().BeEquivalentTo(new List<Layer> { 
                firstLayer.BuildWeights().AddInputs(inputs), 
                secondLayer.BuildWeights().AddParentLayer(firstLayer) 
            });
        }

        [Test]
        public void ANN_That_HasNotBeenBuilt_Should_NotRun()
        {
            var ann = A.Create();
            Action act = () => { var _ = ann.Outputs; };

            act.Should().Throw<InvalidOperationException>().WithMessage("ANN must be built before being run");
        }

        [Test]
        public void ANN_That_HasBeenBuilt_Should_Run()
        {
            var activatorFinal = new SigmoidActivator();

            var inputsSize = 2;
            var inputs = V.DenseOfArray(new double[] { 0.1, 0.9 });

            var firstLayerSize = 3;
            var secondLayerSize = 4;
            var firstLayer = Layer.CreateWithRandomWeights(firstLayerSize, inputsSize, 0, 1, _activator);
            var secondLayer = Layer.CreateWithRandomWeights(secondLayerSize, firstLayerSize, 0, 1, activatorFinal);

            var layers = new List<Layer> { firstLayer.Clone(), secondLayer.Clone() };

            var ann = A.Create(layers, inputs).Build();

            Action act = () => { var _ = ann.Outputs; };

            act.Should().NotThrow<InvalidOperationException>();
        }

        /// <summary>
        ///         0.0 -> 0.0            0.0 -> 0.0
        /// 0.5 ->  0.5 -> 0.25
        /// 1.0 ->  1.0 -> 1.0 -> 1.25 -> 0.5 -> 0.625
        ///         1.0 -> 1.0
        /// 0.5 ->  0.5 -> 0.25
        /// 1.0 ->  0.0 -> 0.0 -> 1.25 -> 1.0 -> 1.25    -> 1.875
        /// </summary>
        [Test]
        public void ANN_Runs_Correctly()
        {
            var inputs = V.DenseOfArray(new double[] { 0.5, 1.0 });

            var firstLayerWeights = new double[,]
            {
                { 0, 0.5, 1.0 },
                { 1.0, 0.5, 0 }
            };
            var secondLayerWeights = new double[,]
            {
                { 0, 0.5, 1.0 }
            };

            var firstLayer = Layer.Create(M.DenseOfArray(firstLayerWeights), _activator);
            var secondLayer = Layer.Create(M.DenseOfArray(secondLayerWeights), _activator);


            var layers = new List<Layer> { firstLayer, secondLayer };
            var ann = A.Create(layers, inputs).Build();

            var result = ann.Outputs;

            result.Should().BeEquivalentTo(V.DenseOfArray(new double[] { 1.875 }));
        }
    }
}
