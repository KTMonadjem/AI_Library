using Training.Algorithm.Interface;

namespace ANN.Interface
{
    public interface IANN: ILearner
    {
        public ITrainer Trainer { get; }

        public void Run();

        public void Train();
    }
}
