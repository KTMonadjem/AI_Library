using Common.Maths.ActivationFunction;
using FluentAssertions;

namespace Tests.Common.Maths;

[TestFixture]
public class ActivationDerivativeTests
{
    [TestCase(100)]
    [TestCase(0)]
    [TestCase(-100)]
    public void BinaryDerivative_Should_ReturnCorrectValues(double x)
    {
        var result = new BinaryActivator().Derive(x);
        result.Should().Be(0);
    }

    [TestCase(100)]
    [TestCase(0)]
    [TestCase(-100)]
    public void LinearDerivative_Should_ReturnCorrectValues(double x)
    {
        var result = new LinearActivator().Derive(x);
        result.Should().Be(1);
    }

    [TestCase(100, 1)]
    [TestCase(0, 1)]
    [TestCase(-100, 0)]
    public void ReLuDerivative_Should_ReturnCorrectValues(double x, double y)
    {
        var result = new ReLuActivator().Derive(x);
        result.Should().Be(y);
    }

    [Test]
    public void LeakyReLuDerivative_Should_Throw_ArgumentOutOfRangeException_When_LeakIsNegative()
    {
        Action act = () => new LeakyReLuActivator(-0.1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestCase(100, 1, 0)]
    [TestCase(0, 1, 1)]
    [TestCase(-100, 0.1, 0.1)]
    [TestCase(-100, 0.5, 0.5)]
    public void LeakyReLuDerivative_Should_ReturnCorrectValues(double x, double y, double leak)
    {
        var result = new LeakyReLuActivator(leak).Derive(x);
        result.Should().Be(y);
    }

    [TestCase(0.1)]
    [TestCase(0)]
    [TestCase(-0.1)]
    public void SigmoidDerivative_Should_ReturnCorrectValues(double x)
    {
        var activator = new SigmoidActivator();
        var sigmoid = activator.Activate(x);

        var result = activator.Derive(x);
        result.Should().Be(sigmoid * (1 - sigmoid));
    }

    [Test]
    public void ELuDerivative_Should_Throw_ArgumentOutOfRangeException_When_AlphaIsNegative()
    {
        Action act = () => new ELuActivator(-0.1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestCase(0, 0.1)]
    [TestCase(-10, 0.1)]
    [TestCase(-100, 0.1)]
    [TestCase(-100, 0.5)]
    [TestCase(999, 0.1)]
    public void ELuActivator_Should_Return_CorrectValues(double input, double alpha)
    {
        var activator = new ELuActivator(alpha);
        var result = activator.Activate(input);

        var derivative = activator.Derive(input);

        if (input >= 0)
            derivative.Should().Be(1);
        else
            derivative.Should().Be(result + alpha);
    }

    [TestCase(0.1)]
    [TestCase(0)]
    [TestCase(-0.1)]
    public void TanhDerivative_Should_ReturnCorrectValues(double x)
    {
        var activator = new TanhActivator();
        var tanh = activator.Activate(x);

        var result = activator.Derive(x);
        result.Should().Be(1 - Math.Pow(tanh, 2));
    }

    [TestCase(0.1)]
    [TestCase(0)]
    [TestCase(-0.1)]
    public void SwishDerivative_Should_ReturnCorrectValues(double x)
    {
        var activator = new SwishActivator();
        var sigmoid = new SigmoidActivator().Activate(x);

        var swish = activator.Activate(x);

        var result = activator.Derive(x);
        result.Should().Be(swish + sigmoid * (1 - swish));
    }
}
