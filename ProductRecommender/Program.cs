using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.IO;

namespace MLNET_Recomendation
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            var data = LoadData(mlContext);
            ITransformer model = BuildAndTrainModel(mlContext, data.training);
            EvaluateModel(mlContext, data.test, model);
        }

        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            var path = "../../../Data/purchasedProducts.csv";

            IDataView dataView = mlContext.Data.LoadFromTextFile<ProductPurchase>(path, hasHeader: true, separatorChar: ',');

            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return (split.TrainSet, split.TestSet);

        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "productIdEncoded", inputColumnName: "productId"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "productIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };
            var pipeline = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            return pipeline.Fit(trainingDataView);
        }

        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(prediction);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<ProductPurchase, ProductPurchasePrediction>(model);

            var testInput = new ProductPurchase { userId = 1, productId = 100 };

            var productPrediction= predictionEngine.Predict(testInput);

            Console.WriteLine(productPrediction);
        }
    }
}
