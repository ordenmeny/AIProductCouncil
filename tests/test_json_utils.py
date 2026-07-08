from ai_product_council.json_utils import clean_llm_text, extract_json_object, extract_question_from_text


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


def test_clean_llm_text_removes_reasoning_markers():
    raw = """
    Thinking Process:
    Role: Product Manager
    Analyze the Request:
    Для MVP стоит начать с каталога шрифтов и покупки лицензии.
    """

    assert clean_llm_text(raw) == "Для MVP стоит начать с каталога шрифтов и покупки лицензии."


def test_extract_question_rejects_reasoning_text():
    raw = "Thinking Process: Role: PM. Constraints: MVP? Какой сценарий важнее?"

    assert extract_question_from_text(raw) == ""
