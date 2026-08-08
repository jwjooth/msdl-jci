from .mlp_bi_rate import predict_bi_rate


def predict_macro(data):
    bi_rate_prediction, err = predict_bi_rate(data)
    if err is None:
        return None
    return bi_rate_prediction,
