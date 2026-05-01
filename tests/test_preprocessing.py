from API.preprocessing_utils import PreprocessText


def test_preprocess_basic():
    text = "Hello!!! 123 https://test.com"

    result = PreprocessText(text)

    assert isinstance(result, str)
    assert result != ""


def test_preprocess_empty():
    result = PreprocessText("")
    assert isinstance(result, str)


def test_preprocess_special_chars():
    text = "@@@###$$$"
    result = PreprocessText(text)

    assert isinstance(result, str)