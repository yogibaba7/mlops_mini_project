from API.model_loading import LoadModel, LoadVector


def test_model_loading():
    model = LoadModel("my_model", "Production")

    assert model is not None


def test_vectorizer_loading():
    vector = LoadVector()

    assert vector is not None


def test_vectorizer_transform():
    vector = LoadVector()

    sample = ["this is good"]

    result = vector.transform(sample)

    assert result.shape[0] == 1