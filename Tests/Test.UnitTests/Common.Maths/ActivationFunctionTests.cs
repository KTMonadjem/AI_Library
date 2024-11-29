using System.ComponentModel;
using Common.Maths.ActivationFunction;
using Common.Maths.ActivationFunction.Helper;
using Common.Maths.ActivationFunction.Interface;
using FluentAssertions;

namespace Tests.Common.Maths;

[TestFixture]
public class ActivatorTests
{
    [TestCase(0, 0)]
    [TestCase(-1, 0)]
    [TestCase(1, 1)]
    [TestCase(999, 1)]
    public void BinaryActivator_Should_Return_CorrectValues(double input, double output)
    {
        var activator = new BinaryActivator();
        var forward = activator.Activate(input);
        forward.Should().Be(output);
        var backward = activator.Derive(input);
        backward.Should().Be(activator.Delta);
    }

    [TestCase(0, 0)]
    [TestCase(-1, -1)]
    [TestCase(1, 1)]
    [TestCase(999, 999)]
    public void LinearActivator_Should_Return_CorrectValues(double input, double output)
    {
        var activator = new LinearActivator();
        var forward = activator.Activate(input);
        forward.Should().Be(output);
        var backward = activator.Derive(input);
        backward.Should().Be(activator.Delta);
    }

    [TestCase(0, 0.5)]
    [TestCase(0.458, 0.61253961)]
    [TestCase(1.6, 0.83201838)]
    [TestCase(-3, 0.04742587)]
    public void SigmoidActivator_Should_Return_CorrectValues(double input, double output)
    {
        var activator = new SigmoidActivator();
        var forward = activator.Activate(input);
        forward.Should().BeApproximately(output, 0.00000001);
        var backward = activator.Derive(input);
        backward.Should().Be(activator.Delta);
    }

    [TestCase(0, 0)]
    [TestCase(2, 0.96402758)]
    [TestCase(-1, -0.76159415)]
    [TestCase(-0.22, -0.21651806)]
    public void TanhActivator_Should_Return_CorrectValues(double input, double output)
    {
        var activator = new TanhActivator();
        var forward = activator.Activate(input);
        forward.Should().BeApproximately(output, 0.00000001);
        var backward = activator.Derive(input);
        backward.Should().Be(activator.Delta);
    }

    [TestCase(0, 0)]
    [TestCase(-1, 0)]
    [TestCase(1, 1)]
    [TestCase(999, 999)]
    public void ReLuActivator_Should_Return_CorrectValues(double input, double output)
    {
        var activator = new ReLuActivator();
        var forward = activator.Activate(input);
        forward.Should().Be(output);
        var backward = activator.Derive(input);
        backward.Should().Be(activator.Delta);
    }

    [TestCase(0, 0, 0.1)]
    [TestCase(-1, -0.1, 0.1)]
    [TestCase(-100, -10, 0.1)]
    [TestCase(-100, -50, 0.5)]
    [TestCase(999, 999, 0.1)]
    public void LeakyReLuActivator_Should_Return_CorrectValues(double input, double output, double leak)
    {
        var activator = new LeakyReLuActivator(leak);
        var forward = activator.Activate(input);
        forward.Should().Be(output);
        var backward = activator.Derive(input);
        backward.Should().Be(activator.Delta);
    }

    [Test]
    public void LeakyReLuActivator_Should_Throw_ArgumentOutOfRangeException_When_LeakIsNegative()
    {
        Action act = () => new LeakyReLuActivator(-0.1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestCase(0, 0, 0.1)]
    [TestCase(-10, -0.09999546, 0.1)]
    [TestCase(-100, -0.1, 0.1)]
    [TestCase(-100, -0.5, 0.5)]
    [TestCase(999, 999, 0.1)]
    public void ELuActivator_Should_Return_CorrectValues(double input, double output, double alpha)
    {
        var activator = new ELuActivator(alpha);
        var forward = activator.Activate(input);
        forward.Should().BeApproximately(output, 0.00000001);
        var backward = activator.Derive(input);
        backward.Should().Be(activator.Delta);
    }

    [Test]
    public void ELuActivator_Should_Throw_ArgumentOutOfRangeException_When_AlphaIsNegative()
    {
        Action act = () => new ELuActivator(-0.1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestCase(0, 0)]
    [TestCase(0.458, 0.28054314)]
    [TestCase(1.6, 1.33122941)]
    [TestCase(-3, -0.14227761)]
    public void SwishActivator_Should_Return_CorrectValues(double input, double output)
    {
        var activator = new SwishActivator();
        var forward = activator.Activate(input);
        forward.Should().BeApproximately(output, 0.00000001);
        var backward = activator.Derive(input);
        backward.Should().Be(activator.Delta);
    }

    [Test]
    public void ActivationMapper_Should_ThrowError_When_InvalidActivationFunction()
    {
        Action act = () => ActivationFunctionMapper.MapActivationFunction((IActivationFunction.ActivationFunction)99);

        act.Should().Throw<InvalidEnumArgumentException>();
    }

    [TestCase(0.0)]
    public void ActivationMapper_Should_MapActivationFunction(double? ctorParam)
    {
    }
}