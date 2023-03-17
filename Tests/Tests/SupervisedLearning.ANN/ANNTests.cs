using ANN.Structure.Layer;
using Common.Maths.ActivationFunction.Interface;
using Common.Maths.ActivationFunction;
using FluentAssertions;
using MathNet.Numerics.LinearAlgebra;
using A = ANN.ANN;

namespace Tests.SupervisedLearning.ANN
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
        public void Build_Should_Succeed_When_ANNIsSetup()
        {

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
            var ann = A.Create(_layers, _inputs).Build();
            Action act = () => { var _ = ann.Outputs; };

            act.Should().NotThrow<InvalidOperationException>();
        }
    }
}
