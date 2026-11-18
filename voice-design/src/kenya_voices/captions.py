"""Turn dataset metadata + measured audio features into voice-design captions.

Voice design in VoxCPM2 is **prompt-conditioned generation**: the description sits
in parentheses at the start of the text and conditions the synthesised voice
(``"(A warm Kenyan male voice in his thirties)Habari yako?"``). Fine-tuning on
caption-prefixed transcripts therefore teaches the model to bind a *described*
Kenyan voice to audio - which is what "let users design Kenyan voices" needs.

Design rules that matter for quality:

* **Only describe what is true.** ``gender`` / ``age_group`` / ``location`` come
  from dataset metadata; pitch and pace are *measured* from the audio (median f0,
  words per second), and code-switching from the transcript's English share.
  Never invent timbre/mood adjectives: unlearnable labels just add noise and make
  the adapter ignore the caption.
* **Vary the wording.** Each row picks a phrasing template deterministically from
  a hash of its id, so the model learns the attributes rather than one sentence.
* **Keep it short.** Long captions eat the 8192-token budget and dilute the loss.

``split_caption`` is the inverse, used by the eval stage.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Attributes we are allowed to state, with the thresholds that decide them.
F0_FEMALE_HZ = 165.0  # median f0 above this -> "female" when metadata is missing
PITCH_BANDS: tuple[tuple[float, str], ...] = (
    (110.0, "low-pitched"),
    (165.0, "mid-pitched"),
    (225.0, "bright, higher-pitched"),
    (10_000.0, "high-pitched"),
)
PACE_BANDS: tuple[tuple[float, str], ...] = (
    (2.2, "measured, unhurried pace"),
    (3.2, "steady conversational pace"),
    (4.2, "brisk city pace"),
    (100.0, "fast, energetic delivery"),
)

_ENGLISH_MARKERS = frozenset(
    """
    the and you we is are to of in on for with that this it my your just like so but
    because before after now today people time good bad very really actually okay yeah
    man guys business government school money job work home phone meeting budget
    deadline apply documents scholarship honestly speaking
    """.split()
)

_AGE_WORDS = {
    "18-24": "in his early twenties",
    "25-35": "in his thirties",
    "36-49": "in his forties",
    "50+": "in his fifties or older",
}
_AGE_WORDS_F = {
    "18-24": "in her early twenties",
    "25-35": "in her thirties",
    "36-49": "in her forties",
    "50+": "in her fifties or older",
}

# Phrasing templates. ``{gender}`` etc. are filled from the attributes above.
TEMPLATES: tuple[str, ...] = (
    "A Kenyan {gender} voice {age}, {pitch}, {pace}, speaking {language}",
    "{gender} Kenyan speaker {age}, {pitch} voice, {pace}, {language}",
    "A {pitch} Kenyan {gender} voice {age} with a {pace}, speaking {language}",
    "Kenyan {gender} {age}, {pitch}, {pace}, natural {language} pronunciation",
    "An authentic Kenyan {gender} voice, {age}, {pitch}, {pace}, {language} speech",
    "{gender} voice from Kenya {age}, {pitch} and {pace}, {language}",
    "{gender} voice from {location}, Kenya, {age}, {pitch}, {pace}, {language}",
    "A Kenyan {gender} from {location} {age}, {pitch}, {pace}, speaking {language}",
)

# Slots filled by build_caption, in the order they appear in the templates.
SLOTS = ("gender", "age", "location", "pitch", "pace", "language")


@dataclass(frozen=True)
class VoiceAttrs:
    gender: str = ""  # "male" | "female" | "" (unknown)
    age: str = ""  # metadata band, e.g. "25-35", or ""
    f0_hz: float | None = None
    words_per_sec: float | None = None
    language: str = "Swahili"
    english_share: float = 0.0
    location: str = ""  # city name from metadata, e.g. "Nairobi"
    country: str = "Kenya"

    @property
    def resolved_gender(self) -> str:
        if self.gender:
            return self.gender
        if self.f0_hz and self.f0_hz > 0:
            return "female" if self.f0_hz >= F0_FEMALE_HZ else "male"
        return ""


def pitch_descriptor(f0_hz: float | None) -> str:
    if not f0_hz or f0_hz <= 0:
        return ""
    for limit, label in PITCH_BANDS:
        if f0_hz < limit:
            return label
    return PITCH_BANDS[-1][1]


def pace_descriptor(words_per_sec: float | None) -> str:
    if not words_per_sec or words_per_sec <= 0:
        return ""
    for limit, label in PACE_BANDS:
        if words_per_sec < limit:
            return label
    return PACE_BANDS[-1][1]


def language_descriptor(language: str, english_share: float = 0.0, swahili_share: float = 0.0) -> str:
    """Code-switching is measurable, so it is safe to describe."""
    if english_share >= 0.25 and swahili_share >= 0.05:
        return "Swahili mixed with English (Sheng-style code-switching)"
    if english_share >= 0.05:
        return f"{language} with some English words"
    return language


def english_share(text: str) -> float:
    tokens = [t.strip(".,!?;:\"'()").lower() for t in (text or "").split()]
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in _ENGLISH_MARKERS) / len(tokens)


def _stable_index(key: str, modulo: int) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _clean_location(location: str) -> str:
    """City names only: drop punctuation and anything that is not a short place name."""
    text = " ".join((location or "").split()).strip(" ,.;")
    if not text or len(text) > 24:
        return ""
    if any(ch.isdigit() for ch in text):
        return ""
    return text.title()


def build_caption(attrs: VoiceAttrs, key: str = "", variant: int = 0) -> str:
    """Short, truthful, varied description WITHOUT the surrounding parentheses.

    ``variant`` rotates the phrasing template; it is what makes a multi-caption
    augmentation of the same clip possible without repeating a sentence.
    """
    gender = attrs.resolved_gender
    age_words = ""
    if attrs.age:
        table = _AGE_WORDS_F if gender == "female" else _AGE_WORDS
        age_words = table.get(attrs.age, "")

    parts = {
        "gender": gender or "speaker",
        "age": age_words or "",
        "location": _clean_location(attrs.location),
        "pitch": pitch_descriptor(attrs.f0_hz) or "",
        "pace": pace_descriptor(attrs.words_per_sec) or "",
        "language": language_descriptor(attrs.language, attrs.english_share),
    }
    template = TEMPLATES[(_stable_index(key or f"{gender}{attrs.age}{attrs.f0_hz}", len(TEMPLATES)) + variant) % len(TEMPLATES)]

    # Drop empty slots, then collapse duplicated/awkward separators.
    filled = template
    for slot in SLOTS:
        filled = filled.replace("{" + slot + "}", parts[slot])
    chunks: list[str] = []
    for chunk in filled.split(","):
        chunk = " ".join(chunk.split()).strip()
        if chunk and chunk.lower() not in {c.lower() for c in chunks}:
            chunks.append(chunk)
    caption = ", ".join(chunks)
    return caption[:1].upper() + caption[1:]


def caption_with_text(caption: str, text: str) -> str:
    """The exact string a training row's ``text`` field must contain."""
    caption = (caption or "").strip()
    if not caption:
        return text
    return f"({caption.rstrip('.')}){text}"


def split_caption(oriented_text: str) -> tuple[str, str]:
    """``"(A Kenyan male voice)Habari"`` -> ``("A Kenyan male voice", "Habari")``."""
    raw = (oriented_text or "").lstrip()
    if raw.startswith("("):
        end = raw.find(")")
        if end > 0:
            return raw[1:end].strip(), raw[end + 1 :].lstrip()
    return "", raw


# ----------------------------------------------------------------- profiles
# A voice-design attribute (pitch, pace) is a property of the *speaker*, not of
# one clip. Aggregating per speaker keeps the caption consistent across all of
# a speaker's clips, which is exactly the signal a design model needs: "this
# description" -> "this voice", every time.

@dataclass(frozen=True)
class SpeakerProfile:
    speaker: str
    gender: str = ""
    age: str = ""
    location: str = ""
    f0_hz: float | None = None
    words_per_sec: float | None = None
    english_share: float = 0.0
    clips: int = 0
    seconds: float = 0.0
    caption: str = ""
    val: bool = False

    def attrs(self) -> VoiceAttrs:
        return VoiceAttrs(
            gender=self.gender,
            age=self.age,
            f0_hz=self.f0_hz,
            words_per_sec=self.words_per_sec,
            english_share=self.english_share,
            location=self.location,
        )

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v and v > 0)
    if len(vals) < 3:  # too few measurements -> do not guess
        return None
    mid = len(vals) // 2
    return round(vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2, 1)


def build_profile(
    speaker: str,
    *,
    gender: str = "",
    age: str = "",
    location: str = "",
    f0_values: list[float] | None = None,
    texts: list[str] | None = None,
    durations: list[float] | None = None,
    clips: int | None = None,
    val: bool = False,
) -> SpeakerProfile:
    """Aggregate clip-level measurements into one captionable speaker profile."""
    texts = [t for t in (texts or []) if t]
    durations = [float(d) for d in (durations or []) if d]
    seconds = round(sum(durations), 2)
    total_words = sum(len(t.split()) for t in texts)
    wps = round(total_words / seconds, 2) if seconds > 0 and total_words >= 5 else None
    english = round(sum(english_share(t) for t in texts) / len(texts), 3) if texts else 0.0
    attrs = VoiceAttrs(
        gender=gender.lower() if gender else "",
        age=age,
        f0_hz=_median(list(f0_values or [])),
        words_per_sec=wps,
        english_share=english,
        location=location,
    )
    return SpeakerProfile(
        speaker=speaker,
        gender=attrs.resolved_gender,
        age=age,
        location=_clean_location(location),
        f0_hz=attrs.f0_hz,
        words_per_sec=wps,
        english_share=english,
        clips=clips if clips is not None else len(texts),
        seconds=seconds,
        caption=build_caption(attrs, key=speaker),
        val=val,
    )


def caption_variants(attrs: VoiceAttrs, key: str, count: int) -> list[str]:
    """``count`` distinct phrasings of the same truthful description."""
    out: list[str] = []
    for variant in range(max(1, count)):
        caption = build_caption(attrs, key=key, variant=variant)
        if caption and caption not in out:
            out.append(caption)
    return out


def drop_caption(key: str, rate: float = 0.0) -> bool:
    """Deterministic caption dropout.

    Some rows must stay plain transcripts, otherwise the adapter *requires* a
    caption at inference time and the base model's ordinary TTS behaviour is
    destroyed. Deterministic so a rebuild produces the same manifests.
    """
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    digest = hashlib.sha1(f"drop:{key}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10_000) < rate * 10_000
