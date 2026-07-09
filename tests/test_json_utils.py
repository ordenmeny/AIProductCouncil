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


def test_clean_llm_text_rejects_english_reasoning():
    raw = "Okay, I'm trying to define the MVP for a font license website."

    assert clean_llm_text(raw) == ""
    assert extract_question_from_text(raw) == ""


def test_clean_llm_text_rejects_mixed_reasoning_text():
    raw = (
        "Хорошо, нужно структурировать ответ в JSON: summary, risks, insights. "
        "Let me break it down step by step."
    )

    assert clean_llm_text(raw) == ""


def test_clean_llm_text_rejects_cjk_symbols():
    raw = "Хорошо, пользователь 扮演 UX Researcher и спрашивает что делать 下一步."

    assert clean_llm_text(raw) == ""


def test_extract_question_accepts_clean_russian_question():
    raw = "Какие способы оплаты обязательны для первой версии сайта по продаже шрифтов?"

    assert extract_question_from_text(raw) == raw
