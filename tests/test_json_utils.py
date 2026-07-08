from ai_product_council.json_utils import extract_json_object


def test_extract_plain_json_object():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_markdown_json_block():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_inside_text():
    assert extract_json_object('answer: {"a": 1}') == {"a": 1}


def test_extract_first_json_when_model_adds_extra_data():
    assert extract_json_object('{"a": 1}\n{"b": 2}') == {"a": 1}


def test_extract_json_before_trailing_text():
    assert extract_json_object('{"a": 1}\nExplanation after JSON') == {"a": 1}
