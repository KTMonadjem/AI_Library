namespace Learning.Supervised.Training.Algorithm.Interface;

public interface ITrainer
{
    public record TrainingOutput
    {
        public double Loss { get; set; }

        public int Epochs { get; set; }
    }

    public TrainingOutput Train();
}
