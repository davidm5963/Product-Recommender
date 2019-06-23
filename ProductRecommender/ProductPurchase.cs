using System;
using Microsoft.ML.Data;

public class ProductPurchase
{
    [LoadColumn(0)]
    public float userId;
    [LoadColumn(1)]
    public float productId;
    [LoadColumn(2)]
    public float Label;

}
