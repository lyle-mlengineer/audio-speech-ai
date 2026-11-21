from __future__ import annotations

from sauti_sheng.text import (
    SHENG_PROBES,
    charset_report,
    choose_ref_prefix,
    clean_text,
    estimate_fraction,
    f5_text_identity,
    join_sentences,
    read_vocab,
    split_sentences,
    truncate_for_eval,
)


def test_clean_text_normalises_quotes_and_space():
    assert clean_text("  Niaje ‘mazee’  \n\t leo  ") == "Niaje 'mazee' leo"


def test_clean_text_is_lossless_for_apostrophes():
    assert clean_text("ng'ang'ana") == "ng'ang'ana"


def test_split_sentences_keeps_fragments():
    assert split_sentences("Mmmh. Eeeh! Unafanya nini? Sawa.") == [
        "Mmmh.",
        "Eeeh!",
        "Unafanya nini?",
        "Sawa.",
    ]


def test_join_sentences():
    assert join_sentences(["a.", " b. "]) == "a. b."


def test_choose_ref_prefix_is_sentence_aligned_and_short():
    text = "Moja ni hii. Mbili ni hiyo. Tatu ni ile. Nne ni zile. Tano ni hizo."
    choice = choose_ref_prefix(text, 0.5)
    assert choice.text in text
    assert choice.n_sentences < choice.total_sentences
    assert abs(choice.fraction - 0.5) < 0.25


def test_choose_ref_prefix_single_sentence():
    choice = choose_ref_prefix("Niaje mazee.", 0.5)
    assert choice.text == "Niaje mazee."


def test_estimate_fraction_clamps():
    assert estimate_fraction(9, 30) == 0.3
    assert estimate_fraction(0, 0) == 0.5


def test_truncate_for_eval_keeps_it_short():
    text = " ".join(["Hii ni sentensi ndefu sana."] * 20)
    out = truncate_for_eval(text, max_chars=100)
    assert len(out) <= 100
    assert out.endswith(".")


def test_charset_report_finds_missing():
    report = charset_report(["abc", "aé"], set("abc "))
    assert report.unique_chars == 4
    assert report.missing == {"é": 1}
    assert not report.clean


def test_charset_report_counts_digits_and_apostrophes():
    report = charset_report(["M-Pesa 2 na ng'ang'ana"], set("M-Pesa 2 na ng'ang'ana"))
    assert report.digits == 1
    assert report.apostrophes == 2


def test_read_vocab_handles_crlf_and_trailing_newline(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_bytes(b" \na\r\nb\r\n")
    assert read_vocab(path) == [" ", "a", "b"]


def test_probes_are_short_and_use_ascii():
    for probe in SHENG_PROBES:
        assert len(probe) < 170
        assert probe.isascii()


def test_f5_text_identity_is_tri_state():
    assert f5_text_identity("Niaje mazee.") in (True, False, None)
