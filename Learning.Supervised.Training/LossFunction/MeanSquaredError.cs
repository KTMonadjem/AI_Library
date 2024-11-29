using Learning.Supervised.Training.LossFunction.Interface;

namespace Learning.Supervised.Training.LossFunction
{
    public class MeanSquaredError: ILossFunction
    {
        public double CalculateLoss(double[] expected, double[] actual)
        {
            if (expected == null)
            {
                throw new ArgumentNullException(nameof(expected));
            }
            if (actual == null)
            {
                throw new ArgumentNullException(nameof(actual));
            }
            if (actual.Length != expected.Length)
            {
                throw new ArgumentException("Expected and actual should be the same length.");
            }

            var sum = 0d;
            for (var i = 0; i < actual.Length; i++)
            {
                sum += CalculateLoss(expected[i], actual[i]);
            }
            return sum / actual.Length;
        }

        public double CalculateLoss(double expected, double actual)
        {
            return Math.Sqrt(Math.Pow(actual - expected, 2));
        }
    }
}
