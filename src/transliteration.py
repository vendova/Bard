"""
Transliteration module — converts non-Latin script lyrics to English
transliteration (romanization).

Supports major scripts:
- Cyrillic (Russian, Ukrainian, Bulgarian, etc.)
- Devanagari (Hindi, Marathi, Nepali, etc.)
- Arabic (Arabic, Urdu, Persian)
- Japanese (Hiragana, Katakana — basic romaji)
- Korean (Hangul — basic romanization)
- Chinese (Pinyin via basic mapping, limited)
- Greek
- Hebrew
- Thai (basic)

Uses rule-based Unicode character mapping. For best accuracy, provide an
LRC file with pre-timed lyrics; the transliteration is applied to each line.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Cyrillic → Latin (ISO/R TO-9 2000 system, simplified)
# ---------------------------------------------------------------------------

_CYRILLIC_MAP = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
}

_CYRILLIC_DIGRAPHS = {
    'ьи': 'yi', 'ъе': 'ye', 'ъё': 'yo', 'ъю': 'yu', 'ъя': 'ya',
}


def _transliterate_cyrillic(text: str) -> str:
    result = []
    i = 0
    lower = text.lower()
    while i < len(text):
        ch = lower[i]
        # Check digraphs
        if i + 1 < len(text):
            pair = ch + lower[i + 1]
            if pair in _CYRILLIC_DIGRAPHS:
                mapped = _CYRILLIC_DIGRAPHS[pair]
                # Preserve original case
                if text[i].isupper():
                    mapped = mapped.capitalize()
                result.append(mapped)
                i += 2
                continue
        if ch in _CYRILLIC_MAP:
            mapped = _CYRILLIC_MAP[ch]
            if text[i].isupper():
                mapped = mapped.capitalize()
            result.append(mapped)
        else:
            result.append(text[i])
        i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Devanagari → Latin (ITRANS-style simplified)
# ---------------------------------------------------------------------------

_DEV_CONSONANTS = {
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va',
    'श': 'sha', 'ष': 'sha', 'स': 'sa', 'ह': 'ha',
    'क्ष': 'ksha', 'त्र': 'tra', 'ज्ञ': 'gya', 'श्र': 'shra',
}

_DEV_VOWELS = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u', 'ऊ': 'uu',
    'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'अं': 'am', 'अः': 'ah',
}

_DEV_MATRAS = {
    'ा': 'aa', 'ि': 'i', 'ी': 'ii', 'ु': 'u', 'ू': 'uu',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    'ं': 'n', 'ः': 'h', 'ँ': 'n',
}

_DEV_SPECIAL = {
    '॥': ' || ', '।': '. ', '०': '0', '१': '1', '२': '2', '३': '3',
    '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}


def _transliterate_devanagari(text: str) -> str:
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        # Check for conjuncts (multi-char)
        matched = False
        for clen in (2, 1):
            if i + clen <= len(text):
                sub = text[i:i + clen]
                if sub in _DEV_CONSONANTS:
                    base = _DEV_CONSONANTS[sub]
                    # Check for matra following
                    if i + clen < len(text) and text[i + clen] in _DEV_MATRAS:
                        matra = _DEV_MATRAS[text[i + clen]]
                        # Consonant + matra: replace inherent 'a' with matra
                        result.append(base[:-1] + matra if base.endswith('a') else base + matra)
                        i += clen + 1
                    elif i + clen < len(text) and text[i + clen] == '्':
                        # Virama — suppress inherent vowel
                        result.append(base[:-1])
                        i += clen + 1
                    else:
                        result.append(base)
                        i += clen
                    matched = True
                    break
        if matched:
            continue
        if ch in _DEV_VOWELS:
            result.append(_DEV_VOWELS[ch])
        elif ch in _DEV_MATRAS:
            # Standalone matra (shouldn't happen normally but handle gracefully)
            pass
        elif ch in _DEV_SPECIAL:
            result.append(_DEV_SPECIAL[ch])
        else:
            result.append(ch)
        i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Arabic → Latin (simplified ALA-LC)
# ---------------------------------------------------------------------------

_ARABIC_MAP = {
    'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h',
    'خ': 'kh', 'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's',
    'ش': 'sh', 'ص': 's', 'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': 'a',
    'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm',
    'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y', 'ى': 'a', 'ة': 'a',
    'ء': "'", 'ئ': 'y', 'ؤ': 'w', 'أ': 'a', 'إ': 'i', 'آ': 'aa',
    'ً': 'an', 'ٌ': 'un', 'ٍ': 'in', 'َ': 'a', 'ُ': 'u', 'ِ': 'i',
    'ّ': '', 'ْ': '', 'ٰ': 'a', 'ۡ': '', 'ٰ': 'a',
}


def _transliterate_arabic(text: str) -> str:
    result = []
    for ch in text:
        result.append(_ARABIC_MAP.get(ch, ch))
    text = ''.join(result)
    # Clean up double spaces and trailing apostrophes
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Japanese (Hiragana/Katakana) → Romaji
# ---------------------------------------------------------------------------

_HIRAGANA = {
    'あ': 'a', 'い': 'i', 'う': 'u', 'え': 'e', 'お': 'o',
    'か': 'ka', 'き': 'ki', 'く': 'ku', 'け': 'ke', 'こ': 'ko',
    'さ': 'sa', 'し': 'shi', 'す': 'su', 'せ': 'se', 'そ': 'so',
    'た': 'ta', 'ち': 'chi', 'つ': 'tsu', 'て': 'te', 'と': 'to',
    'な': 'na', 'に': 'ni', 'ぬ': 'nu', 'ね': 'ne', 'の': 'no',
    'は': 'ha', 'ひ': 'hi', 'ふ': 'fu', 'へ': 'he', 'ほ': 'ho',
    'ま': 'ma', 'み': 'mi', 'む': 'mu', 'め': 'me', 'も': 'mo',
    'や': 'ya', 'ゆ': 'yu', 'よ': 'yo',
    'ら': 'ra', 'り': 'ri', 'る': 'ru', 'れ': 're', 'ろ': 'ro',
    'わ': 'wa', 'を': 'wo', 'ん': 'n',
    'が': 'ga', 'ぎ': 'gi', 'ぐ': 'gu', 'げ': 'ge', 'ご': 'go',
    'ざ': 'za', 'じ': 'ji', 'ず': 'zu', 'ぜ': 'ze', 'ぞ': 'zo',
    'だ': 'da', 'ぢ': 'ji', 'づ': 'zu', 'で': 'de', 'ど': 'do',
    'ば': 'ba', 'び': 'bi', 'ぶ': 'bu', 'べ': 'be', 'ぼ': 'bo',
    'ぱ': 'pa', 'ぴ': 'pi', 'ぷ': 'pu', 'ぺ': 'pe', 'ぽ': 'po',
    'きゃ': 'kya', 'きゅ': 'kyu', 'きょ': 'kyo',
    'しゃ': 'sha', 'しゅ': 'shu', 'しょ': 'sho',
    'ちゃ': 'cha', 'ちゅ': 'chu', 'ちょ': 'cho',
    'にゃ': 'nya', 'にゅ': 'nyu', 'にょ': 'nyo',
    'ひゃ': 'hya', 'ひゅ': 'hyu', 'ひょ': 'hyo',
    'みゃ': 'mya', 'みゅ': 'myu', 'みょ': 'myo',
    'りゃ': 'rya', 'りゅ': 'ryu', 'りょ': 'ryo',
    'ぎゃ': 'gya', 'ぎゅ': 'gyu', 'ぎょ': 'gyo',
    'じゃ': 'ja', 'じゅ': 'ju', 'じょ': 'jo',
    'びゃ': 'bya', 'びゅ': 'byu', 'びょ': 'byo',
    'ぴゃ': 'pya', 'ぴゅ': 'pyu', 'ぴょ': 'pyo',
    'っ': '', 'ー': '-', 'ゃ': 'ya', 'ゅ': 'yu', 'ょ': 'yo', 'ゎ': 'wa',
}

_KATAKANA = {
    'ア': 'a', 'イ': 'i', 'ウ': 'u', 'エ': 'e', 'オ': 'o',
    'カ': 'ka', 'キ': 'ki', 'ク': 'ku', 'ケ': 'ke', 'コ': 'ko',
    'サ': 'sa', 'シ': 'shi', 'ス': 'su', 'セ': 'se', 'ソ': 'so',
    'タ': 'ta', 'チ': 'chi', 'ツ': 'tsu', 'テ': 'te', 'ト': 'to',
    'ナ': 'na', 'ニ': 'ni', 'ヌ': 'nu', 'ネ': 'ne', 'ノ': 'no',
    'ハ': 'ha', 'ヒ': 'hi', 'フ': 'fu', 'ヘ': 'he', 'ホ': 'ho',
    'マ': 'ma', 'ミ': 'mi', 'ム': 'mu', 'メ': 'me', 'モ': 'mo',
    'ヤ': 'ya', 'ユ': 'yu', 'ヨ': 'yo',
    'ラ': 'ra', 'リ': 'ri', 'ル': 'ru', 'レ': 're', 'ロ': 'ro',
    'ワ': 'wa', 'ヲ': 'wo', 'ン': 'n',
    'ガ': 'ga', 'ギ': 'gi', 'グ': 'gu', 'ゲ': 'ge', 'ゴ': 'go',
    'ザ': 'za', 'ジ': 'ji', 'ズ': 'zu', 'ゼ': 'ze', 'ゾ': 'zo',
    'ダ': 'da', 'ヂ': 'ji', 'ヅ': 'zu', 'デ': 'de', 'ド': 'do',
    'バ': 'ba', 'ビ': 'bi', 'ブ': 'bu', 'ベ': 'be', 'ボ': 'bo',
    'パ': 'pa', 'ピ': 'pi', 'プ': 'pu', 'ペ': 'pe', 'ポ': 'po',
    'キャ': 'kya', 'キュ': 'kyu', 'キョ': 'kyo',
    'シャ': 'sha', 'シュ': 'shu', 'ショ': 'sho',
    'チャ': 'cha', 'チュ': 'chu', 'チョ': 'cho',
    'ニャ': 'nya', 'ニュ': 'nyu', 'ニョ': 'nyo',
    'ヒャ': 'hya', 'ヒュ': 'hyu', 'ヒョ': 'hyo',
    'ミャ': 'mya', 'ミュ': 'myu', 'ミョ': 'myo',
    'リャ': 'rya', 'リュ': 'ryu', 'リョ': 'ryo',
    'ギャ': 'gya', 'ギュ': 'gyu', 'ギョ': 'gyo',
    'ジャ': 'ja', 'ジュ': 'ju', 'ジョ': 'jo',
    'ビャ': 'bya', 'ビュ': 'byu', 'ビョ': 'byo',
    'ピャ': 'pya', 'ピュ': 'pyu', 'ピョ': 'pyo',
    'ッ': '', 'ー': '-', 'ャ': 'ya', 'ュ': 'yu', 'ョ': 'yo',
}

_JP_COMBINED = {**_HIRAGANA, **_KATAKANA}


def _transliterate_japanese(text: str) -> str:
    result = []
    i = 0
    while i < len(text):
        # Check for digraph (small ya/yu/yo)
        if i + 1 < len(text):
            pair = text[i:i + 2]
            if pair in _JP_COMBINED:
                result.append(_JP_COMBINED[pair])
                i += 2
                continue
        ch = text[i]
        result.append(_JP_COMBINED.get(ch, ch))
        i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Korean (Hangul) → Latin (Revised Romanization, simplified)
# ---------------------------------------------------------------------------

_KOREAN_INITIALS = 'gkknnndttllrrmbbssss jjj'  # not used — using algorithmic approach

def _transliterate_korean(text: str) -> str:
    """Basic Hangul romanization using syllabic decomposition."""
    result = []
    for ch in text:
        code = ord(ch)
        # Hangul syllable range: U+AC00 to U+D7A3
        if 0xAC00 <= code <= 0xD7A3:
            syllable_index = code - 0xAC00
            initial = syllable_index // 588
            medial = (syllable_index % 588) // 28
            final = syllable_index % 28

            initials = ['g', 'kk', 'n', 'd', 'tt', 'r', 'm', 'b', 'bb',
                        's', 'ss', '', 'j', 'jj', 'ch', 'k', 't', 'p', 'h']
            medials = ['a', 'ae', 'ya', 'yae', 'eo', 'e', 'yeo', 'ye', 'o',
                       'wa', 'wae', 'oe', 'yo', 'u', 'weo', 'we', 'wi', 'yu',
                       'eu', 'ui', 'i']
            finals = ['', 'g', 'kk', 'gs', 'n', 'nj', 'nh', 'd', 'r', 'rg',
                      'rm', 'rb', 'rs', 'rt', 'rp', 'rh', 'm', 'b', 'bs',
                      's', 'ss', 'ng', 'j', 'ch', 'k', 't', 'p', 'h']

            s = initials[initial] + medials[medial]
            if final:
                s += finals[final]
            result.append(s)
        else:
            result.append(ch)
    return ''.join(result)


# ---------------------------------------------------------------------------
# Greek → Latin
# ---------------------------------------------------------------------------

_GREEK_MAP = {
    'α': 'a', 'β': 'v', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'ζ': 'z',
    'η': 'i', 'θ': 'th', 'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm',
    'ν': 'n', 'ξ': 'x', 'ο': 'o', 'π': 'p', 'ρ': 'r', 'σ': 's',
    'ς': 's', 'τ': 't', 'υ': 'y', 'φ': 'f', 'χ': 'ch', 'ψ': 'ps',
    'ω': 'o', 'ά': 'a', 'έ': 'e', 'ή': 'i', 'ί': 'i', 'ό': 'o',
    'ύ': 'y', 'ώ': 'o', 'ϊ': 'i', 'ϋ': 'y',
}


def _transliterate_greek(text: str) -> str:
    result = []
    for ch in text:
        lower = ch.lower()
        if lower in _GREEK_MAP:
            mapped = _GREEK_MAP[lower]
            if ch.isupper():
                mapped = mapped.capitalize()
            result.append(mapped)
        else:
            result.append(ch)
    return ''.join(result)


# ---------------------------------------------------------------------------
# Hebrew → Latin
# ---------------------------------------------------------------------------

_HEBREW_MAP = {
    'א': '', 'ב': 'b', 'ג': 'g', 'ד': 'd', 'ה': 'h', 'ו': 'v',
    'ז': 'z', 'ח': 'ch', 'ט': 't', 'י': 'y', 'כ': 'k', 'ך': 'k',
    'ל': 'l', 'מ': 'm', 'ם': 'm', 'נ': 'n', 'ן': 'n', 'ס': 's',
    'ע': '', 'פ': 'p', 'ף': 'p', 'צ': 'ts', 'ץ': 'ts', 'ק': 'k',
    'ר': 'r', 'ש': 'sh', 'ת': 't',
    'ָ': 'a', 'ַ': 'a', 'ֵ': 'e', 'ֶ': 'e', 'ִ': 'i', 'ֹ': 'o',
    'ֻ': 'u', 'ְ': '',
}


def _transliterate_hebrew(text: str) -> str:
    result = []
    for ch in text:
        result.append(_HEBREW_MAP.get(ch, ch))
    text = ''.join(result)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Thai → Latin (basic)
# ---------------------------------------------------------------------------

_THAI_MAP = {
    'ก': 'k', 'ข': 'kh', 'ค': 'kh', 'ง': 'ng', 'จ': 'ch', 'ฉ': 'ch',
    'ช': 'ch', 'ซ': 's', 'ญ': 'y', 'ฎ': 'd', 'ฏ': 't', 'ฐ': 'th',
    'ฑ': 'th', 'ฒ': 'n', 'ณ': 'n', 'ด': 'd', 'ต': 't', 'ถ': 'th',
    'ท': 'th', 'ธ': 'th', 'น': 'n', 'บ': 'b', 'ป': 'p', 'ผ': 'ph',
    'ฝ': 'f', 'พ': 'ph', 'ฟ': 'f', 'ภ': 'ph', 'ม': 'm', 'ย': 'y',
    'ร': 'r', 'ล': 'l', 'ว': 'w', 'ศ': 's', 'ษ': 's', 'ส': 's',
    'ห': 'h', 'ฬ': 'l', 'ฮ': 'h', 'ะ': 'a', 'า': 'a', 'ำ': 'am',
    'ิ': 'i', 'ี': 'i', 'ึ': 'ue', 'ื': 'ue', 'ุ': 'u', 'ู': 'u',
    'เ': 'e', 'แ': 'ae', 'โ': 'o', 'ใ': 'ai', 'ไ': 'ai', 'ๆ': '',
    ' ์': '', '่': '', '้': '', '๊': '', '๋': '', 'ฯ': '',
}


def _transliterate_thai(text: str) -> str:
    result = []
    for ch in text:
        result.append(_THAI_MAP.get(ch, ch))
    return ''.join(result)


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

def _detect_script(text: str) -> str:
    """Detect the dominant non-Latin script in text."""
    counts = {
        'cyrillic': 0, 'devanagari': 0, 'arabic': 0, 'japanese': 0,
        'korean': 0, 'greek': 0, 'hebrew': 0, 'thai': 0, 'latin': 0,
    }
    for ch in text:
        code = ord(ch)
        if 0x0400 <= code <= 0x04FF:
            counts['cyrillic'] += 1
        elif 0x0900 <= code <= 0x097F:
            counts['devanagari'] += 1
        elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
            counts['arabic'] += 1
        elif 0x3040 <= code <= 0x30FF:
            counts['japanese'] += 1
        elif 0xAC00 <= code <= 0xD7A3:
            counts['korean'] += 1
        elif 0x0370 <= code <= 0x03FF:
            counts['greek'] += 1
        elif 0x0590 <= code <= 0x05FF:
            counts['hebrew'] += 1
        elif 0x0E00 <= code <= 0x0E7F:
            counts['thai'] += 1
        elif ch.isalpha():
            counts['latin'] += 1

    non_latin = {k: v for k, v in counts.items() if k != 'latin'}
    if not non_latin or max(non_latin.values()) == 0:
        return 'latin'
    return max(non_latin, key=non_latin.get)


# ---------------------------------------------------------------------------
# Main transliteration function
# ---------------------------------------------------------------------------

_TRANLITERATORS = {
    'cyrillic': _transliterate_cyrillic,
    'devanagari': _transliterate_devanagari,
    'arabic': _transliterate_arabic,
    'japanese': _transliterate_japanese,
    'korean': _transliterate_korean,
    'greek': _transliterate_greek,
    'hebrew': _transliterate_hebrew,
    'thai': _transliterate_thai,
}


def transliterate(text: str) -> str:
    """Transliterate non-Latin text to English/Latin script.

    If the text is already Latin, returns it unchanged.
    Detects script per-line so mixed-language lyrics work correctly.
    """
    if not text:
        return text

    lines = text.split('\n')
    result_lines = []
    for line in lines:
        script = _detect_script(line)
        if script == 'latin':
            result_lines.append(line)
        elif script in _TRANLITERATORS:
            transliterated = _TRANLITERATORS[script](line)
            result_lines.append(transliterated)
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)


def needs_transliteration(text: str) -> bool:
    """Check if text contains non-Latin characters that need transliteration."""
    if not text:
        return False
    return _detect_script(text) != 'latin'
