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
# Generic Brahmic-script transliterator
# Used by Telugu, Tamil, Malayalam, Kannada (and could replace Devanagari).
# All Brahmic scripts share the same structure:
#   • Consonants carry an inherent 'a' vowel (ka, kha, ga, …)
#   • Matras (vowel signs) replace the inherent vowel (క + ు = ku, not kau)
#   • Virama suppresses the inherent vowel (క + ് = k)
#   • Consonant clusters use virama between consonants (క్ష = k+sha)
# ---------------------------------------------------------------------------

def _transliterate_brahmic(
    text: str,
    consonants: Dict[str, str],
    vowels: Dict[str, str],
    matras: Dict[str, str],
    specials: Dict[str, str],
    virama: str,
) -> str:
    """Transliterate any Brahmic script using shared structural logic."""
    result: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        # Independent vowel
        if ch in vowels:
            result.append(vowels[ch])
            i += 1
            continue

        # Consonant — check for conjunct (2-char) first
        matched_consonant = False
        for clen in (2, 1):
            if i + clen <= n:
                sub = text[i:i + clen]
                if sub in consonants:
                    base = consonants[sub]
                    next_ch = text[i + clen] if (i + clen < n) else ''
                    if next_ch == virama:
                        # Virama: suppress inherent 'a'
                        # Check if there's a matra after the virama
                        # (rare but possible in some orthographies)
                        result.append(base[:-1] if base.endswith('a') else base)
                        i += clen + 1
                        matched_consonant = True
                        break
                    elif next_ch in matras:
                        # Matra replaces inherent 'a'
                        matra_val = matras[next_ch]
                        if base.endswith('a'):
                            result.append(base[:-1] + matra_val)
                        else:
                            result.append(base + matra_val)
                        i += clen + 1
                        matched_consonant = True
                        break
                    else:
                        # Consonant with inherent 'a'
                        result.append(base)
                        i += clen
                        matched_consonant = True
                        break

        if matched_consonant:
            continue

        # Standalone matra (shouldn't appear without consonant, but handle gracefully)
        if ch in matras:
            result.append(matras[ch])
            i += 1
            continue

        # Virama alone (shouldn't happen, skip it)
        if ch == virama:
            i += 1
            continue

        # Special characters (numerals, punctuation, danda)
        if ch in specials:
            result.append(specials[ch])
            i += 1
            continue

        # Pass through everything else (Latin, punctuation, spaces)
        result.append(ch)
        i += 1

    return ''.join(result)


# ---------------------------------------------------------------------------
# Telugu (U+0C00–U+0C7F)
# ---------------------------------------------------------------------------

_TELUGU_VOWELS = {
    'అ': 'a', 'ఆ': 'aa', 'ఇ': 'i', 'ఈ': 'ii', 'ఉ': 'u', 'ఊ': 'uu',
    'ఋ': 'ru', 'ఎ': 'e', 'ఏ': 'ee', 'ఐ': 'ai', 'ఒ': 'o', 'ఓ': 'oo',
    'ఔ': 'au',
}

_TELUGU_MATRAS = {
    'ా': 'aa', 'ి': 'i', 'ీ': 'ii', 'ు': 'u', 'ూ': 'uu', 'ృ': 'ru',
    'ె': 'e', 'ే': 'ee', 'ై': 'ai', 'ొ': 'o', 'ో': 'oo', 'ౌ': 'au',
}

_TELUGU_CONSONANTS = {
    'క': 'ka', 'ఖ': 'kha', 'గ': 'ga', 'ఘ': 'gha', 'ఙ': 'nga',
    'చ': 'cha', 'ఛ': 'chha', 'జ': 'ja', 'ఝ': 'jha', 'ఞ': 'nya',
    'ట': 'ta', 'ఠ': 'tha', 'డ': 'da', 'ఢ': 'dha', 'ణ': 'na',
    'త': 'ta', 'థ': 'tha', 'ద': 'da', 'ధ': 'dha', 'న': 'na',
    'ప': 'pa', 'ఫ': 'pha', 'బ': 'ba', 'భ': 'bha', 'మ': 'ma',
    'య': 'ya', 'ర': 'ra', 'ఱ': 'ra', 'ల': 'la', 'ళ': 'la',
    'వ': 'va', 'శ': 'sha', 'ష': 'sha', 'స': 'sa', 'హ': 'ha',
    'క్ష': 'ksha', 'త్ర': 'tra', 'జ్ఞ': 'gya',
}

_TELUGU_SPECIALS = {
    'ం': 'n', 'ః': 'h', 'ఁ': 'n',
    '౦': '0', '౧': '1', '౨': '2', '౩': '3', '౪': '4',
    '౫': '5', '౬': '6', '౭': '7', '౮': '8', '౯': '9',
    '।': '. ', '॥': ' || ',
}

_TELUGU_VIRAMA = '్'


def _transliterate_telugu(text: str) -> str:
    return _transliterate_brahmic(text, _TELUGU_CONSONANTS, _TELUGU_VOWELS,
                                  _TELUGU_MATRAS, _TELUGU_SPECIALS, _TELUGU_VIRAMA)


# ---------------------------------------------------------------------------
# Tamil (U+0B80–U+0BFF)
# ---------------------------------------------------------------------------

_TAMIL_VOWELS = {
    'அ': 'a', 'ஆ': 'aa', 'இ': 'i', 'ஈ': 'ii', 'உ': 'u', 'ஊ': 'uu',
    'எ': 'e', 'ஏ': 'ee', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'oo', 'ஔ': 'au',
    'ஃ': 'h',   # aytham
}

_TAMIL_MATRAS = {
    'ா': 'aa', 'ி': 'i', 'ீ': 'ii', 'ு': 'u', 'ூ': 'uu',
    'ெ': 'e', 'ே': 'ee', 'ை': 'ai', 'ொ': 'o', 'ோ': 'oo', 'ௌ': 'au',
}

_TAMIL_CONSONANTS = {
    'க': 'ka', 'ங': 'nga', 'ச': 'cha', 'ஞ': 'nya',
    'ட': 'ta', 'ண': 'na', 'த': 'ta', 'ந': 'na', 'ன': 'na',
    'ப': 'pa', 'ம': 'ma', 'ய': 'ya', 'ர': 'ra', 'ற': 'ra',
    'ல': 'la', 'ள': 'la', 'ழ': 'zha', 'வ': 'va',
    'ஶ': 'sha', 'ஷ': 'sha', 'ஸ': 'sa', 'ஹ': 'ha', 'ஜ': 'ja',
    'க்ஷ': 'ksha', 'த்ர': 'tra', 'ஜ்ஞ': 'gya',
}

_TAMIL_SPECIALS = {
    'ஂ': 'n', 'ஃ': 'h',
    '௦': '0', '௧': '1', '௨': '2', '௩': '3', '௪': '4',
    '௫': '5', '௬': '6', '௭': '7', '௮': '8', '௯': '9',
    '।': '. ', '॥': ' || ',
}

_TAMIL_VIRAMA = '்'   # pulli


def _transliterate_tamil(text: str) -> str:
    return _transliterate_brahmic(text, _TAMIL_CONSONANTS, _TAMIL_VOWELS,
                                  _TAMIL_MATRAS, _TAMIL_SPECIALS, _TAMIL_VIRAMA)


# ---------------------------------------------------------------------------
# Malayalam (U+0D00–U+0D7F)
# ---------------------------------------------------------------------------

_MALAYALAM_VOWELS = {
    'അ': 'a', 'ആ': 'aa', 'ഇ': 'i', 'ഈ': 'ii', 'ഉ': 'u', 'ഊ': 'uu',
    'ഋ': 'ru', 'എ': 'e', 'ഏ': 'ee', 'ഐ': 'ai', 'ഒ': 'o', 'ഓ': 'oo',
    'ഔ': 'au',
}

_MALAYALAM_MATRAS = {
    'ാ': 'aa', 'ി': 'i', 'ീ': 'ii', 'ു': 'u', 'ൂ': 'uu', 'ൃ': 'ru',
    'െ': 'e', 'േ': 'ee', 'ൈ': 'ai', 'ൊ': 'o', 'ോ': 'oo', 'ൌ': 'au',
}

_MALAYALAM_CONSONANTS = {
    'ക': 'ka', 'ഖ': 'kha', 'ഗ': 'ga', 'ഘ': 'gha', 'ങ': 'nga',
    'ച': 'cha', 'ഛ': 'chha', 'ജ': 'ja', 'ഝ': 'jha', 'ഞ': 'nya',
    'ട': 'ta', 'ഠ': 'tha', 'ഡ': 'da', 'ഢ': 'dha', 'ണ': 'na',
    'ത': 'ta', 'ഥ': 'tha', 'ദ': 'da', 'ധ': 'dha', 'ന': 'na',
    'പ': 'pa', 'ഫ': 'pha', 'ബ': 'ba', 'ഭ': 'bha', 'മ': 'ma',
    'യ': 'ya', 'ര': 'ra', 'റ': 'ra', 'ല': 'la', 'ള': 'la',
    'ഴ': 'zha', 'വ': 'va', 'ശ': 'sha', 'ഷ': 'sha', 'സ': 'sa', 'ഹ': 'ha',
    'ക്ഷ': 'ksha', 'ത്ര': 'tra', 'ജ്ഞ': 'gya',
}

_MALAYALAM_SPECIALS = {
    'ം': 'n', 'ഃ': 'h', 'ഁ': 'n',
    # Chillu letters (consonant without inherent vowel)
    'ൻ': 'n', 'ൺ': 'n', 'ർ': 'r', 'ൽ': 'l', 'ൾ': 'l',
    'ൿ': 'k', 'ൔ': 'm', 'ൕ': 'y', 'ൖ': 'l',
    '൦': '0', '൧': '1', '൨': '2', '൩': '3', '൪': '4',
    '൫': '5', '൬': '6', '൭': '7', '൮': '8', '൯': '9',
    '।': '. ', '॥': ' || ',
}

_MALAYALAM_VIRAMA = '്'


def _transliterate_malayalam(text: str) -> str:
    return _transliterate_brahmic(text, _MALAYALAM_CONSONANTS, _MALAYALAM_VOWELS,
                                  _MALAYALAM_MATRAS, _MALAYALAM_SPECIALS, _MALAYALAM_VIRAMA)


# ---------------------------------------------------------------------------
# Kannada (U+0C80–U+0CFF)
# ---------------------------------------------------------------------------

_KANNADA_VOWELS = {
    'ಅ': 'a', 'ಆ': 'aa', 'ಇ': 'i', 'ಈ': 'ii', 'ಉ': 'u', 'ಊ': 'uu',
    'ಋ': 'ru', 'ಎ': 'e', 'ಏ': 'ee', 'ಐ': 'ai', 'ಒ': 'o', 'ಓ': 'oo',
    'ಔ': 'au',
}

_KANNADA_MATRAS = {
    'ಾ': 'aa', 'ಿ': 'i', 'ೀ': 'ii', 'ು': 'u', 'ೂ': 'uu', 'ೃ': 'ru',
    'ೆ': 'e', 'ೇ': 'ee', 'ೈ': 'ai', 'ೊ': 'o', 'ೋ': 'oo', 'ೌ': 'au',
}

_KANNADA_CONSONANTS = {
    'ಕ': 'ka', 'ಖ': 'kha', 'ಗ': 'ga', 'ಘ': 'gha', 'ಙ': 'nga',
    'ಚ': 'cha', 'ಛ': 'chha', 'ಜ': 'ja', 'ಝ': 'jha', 'ಞ': 'nya',
    'ಟ': 'ta', 'ಠ': 'tha', 'ಡ': 'da', 'ಢ': 'dha', 'ಣ': 'na',
    'ತ': 'ta', 'ಥ': 'tha', 'ದ': 'da', 'ಧ': 'dha', 'ನ': 'na',
    'ಪ': 'pa', 'ಫ': 'pha', 'ಬ': 'ba', 'ಭ': 'bha', 'ಮ': 'ma',
    'ಯ': 'ya', 'ರ': 'ra', 'ಱ': 'ra', 'ಲ': 'la', 'ಳ': 'la',
    'ವ': 'va', 'ಶ': 'sha', 'ಷ': 'sha', 'ಸ': 'sa', 'ಹ': 'ha',
    'ಕ್ಷ': 'ksha', 'ತ್ರ': 'tra', 'ಜ್ಞ': 'gya',
}

_KANNADA_SPECIALS = {
    'ಂ': 'n', 'ಃ': 'h', 'ಁ': 'n',
    '೦': '0', '೧': '1', '೨': '2', '೩': '3', '೪': '4',
    '೫': '5', '೬': '6', '೭': '7', '೮': '8', '೯': '9',
    '।': '. ', '॥': ' || ',
}

_KANNADA_VIRAMA = '್'


def _transliterate_kannada(text: str) -> str:
    return _transliterate_brahmic(text, _KANNADA_CONSONANTS, _KANNADA_VOWELS,
                                  _KANNADA_MATRAS, _KANNADA_SPECIALS, _KANNADA_VIRAMA)


# ---------------------------------------------------------------------------
# Bengali (U+0980–U+09FF) — Bengali, Assamese
# ---------------------------------------------------------------------------

_BENGALI_VOWELS = {
    'অ': 'a', 'আ': 'aa', 'ই': 'i', 'ঈ': 'ii', 'উ': 'u', 'ঊ': 'uu',
    'ঋ': 'ri', 'এ': 'e', 'ঐ': 'ai', 'ও': 'o', 'ঔ': 'au',
}

_BENGALI_MATRAS = {
    'া': 'aa', 'ি': 'i', 'ী': 'ii', 'ু': 'u', 'ূ': 'uu', 'ৃ': 'ri',
    'ে': 'e', 'ৈ': 'ai', 'ো': 'o', 'ৌ': 'au',
}

_BENGALI_CONSONANTS = {
    'ক': 'ka', 'খ': 'kha', 'গ': 'ga', 'ঘ': 'gha', 'ঙ': 'nga',
    'চ': 'cha', 'ছ': 'chha', 'জ': 'ja', 'ঝ': 'jha', 'ঞ': 'nya',
    'ট': 'ta', 'ঠ': 'tha', 'ড': 'da', 'ঢ': 'dha', 'ণ': 'na',
    'ত': 'ta', 'থ': 'tha', 'দ': 'da', 'ধ': 'dha', 'ন': 'na',
    'প': 'pa', 'ফ': 'pha', 'ব': 'ba', 'ভ': 'bha', 'ম': 'ma',
    'য': 'ya', 'র': 'ra', 'ল': 'la', 'শ': 'sha', 'ষ': 'sha',
    'স': 'sa', 'হ': 'ha', 'ড়': 'ra', 'ঢ়': 'rha', 'য়': 'ya',
    'ক্ষ': 'ksha', 'ত্র': 'tra', 'জ্ঞ': 'gya',
}

_BENGALI_SPECIALS = {
    'ং': 'n', 'ঃ': 'h', 'ঁ': 'n',
    '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
    '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9',
    '।': '. ', '॥': ' || ',
}

_BENGALI_VIRAMA = '্'


def _transliterate_bengali(text: str) -> str:
    return _transliterate_brahmic(text, _BENGALI_CONSONANTS, _BENGALI_VOWELS,
                                  _BENGALI_MATRAS, _BENGALI_SPECIALS, _BENGALI_VIRAMA)


# ---------------------------------------------------------------------------
# Gurmukhi (U+0A00–U+0A7F) — Punjabi
# ---------------------------------------------------------------------------

_GURMUKHI_VOWELS = {
    'ਅ': 'a', 'ਆ': 'aa', 'ਇ': 'i', 'ਈ': 'ii', 'ਉ': 'u', 'ਊ': 'uu',
    'ਏ': 'e', 'ਐ': 'ai', 'ਓ': 'o', 'ਔ': 'au',
}

_GURMUKHI_MATRAS = {
    'ਾ': 'aa', 'ਿ': 'i', 'ੀ': 'ii', 'ੁ': 'u', 'ੂ': 'uu',
    'ੇ': 'e', 'ੈ': 'ai', 'ੋ': 'o', 'ੌ': 'au',
}

_GURMUKHI_CONSONANTS = {
    'ਕ': 'ka', 'ਖ': 'kha', 'ਗ': 'ga', 'ਘ': 'gha', 'ਙ': 'nga',
    'ਚ': 'cha', 'ਛ': 'chha', 'ਜ': 'ja', 'ਝ': 'jha', 'ਞ': 'nya',
    'ਟ': 'ta', 'ਠ': 'tha', 'ਡ': 'da', 'ਢ': 'dha', 'ਣ': 'na',
    'ਤ': 'ta', 'ਥ': 'tha', 'ਦ': 'da', 'ਧ': 'dha', 'ਨ': 'na',
    'ਪ': 'pa', 'ਫ': 'pha', 'ਬ': 'ba', 'ਭ': 'bha', 'ਮ': 'ma',
    'ਯ': 'ya', 'ਰ': 'ra', 'ਲ': 'la', 'ਵ': 'va', 'ਸ਼': 'sha',
    'ਸ': 'sa', 'ਹ': 'ha', 'ਖ਼': 'kha', 'ਗ਼': 'ga', 'ਜ਼': 'za',
    'ੜ': 'ra', 'ਫ਼': 'pha', 'ਲ਼': 'la',
}

_GURMUKHI_SPECIALS = {
    'ਂ': 'n', 'ਃ': 'h', 'ਁ': 'n', 'ੰ': 'n', 'ੱ': '',
    '੦': '0', '੧': '1', '੨': '2', '੩': '3', '੪': '4',
    '੫': '5', '੬': '6', '੭': '7', '੮': '8', '੯': '9',
    '।': '. ', '॥': ' || ',
}

_GURMUKHI_VIRAMA = '੍'


def _transliterate_gurmukhi(text: str) -> str:
    return _transliterate_brahmic(text, _GURMUKHI_CONSONANTS, _GURMUKHI_VOWELS,
                                  _GURMUKHI_MATRAS, _GURMUKHI_SPECIALS, _GURMUKHI_VIRAMA)


# ---------------------------------------------------------------------------
# Gujarati (U+0A80–U+0AFF)
# ---------------------------------------------------------------------------

_GUJARATI_VOWELS = {
    'અ': 'a', 'આ': 'aa', 'ઇ': 'i', 'ઈ': 'ii', 'ઉ': 'u', 'ઊ': 'uu',
    'ઋ': 'ri', 'એ': 'e', 'ઐ': 'ai', 'ઓ': 'o', 'ઔ': 'au',
}

_GUJARATI_MATRAS = {
    'ા': 'aa', 'િ': 'i', 'ી': 'ii', 'ુ': 'u', 'ૂ': 'uu', 'ૃ': 'ri',
    'ે': 'e', 'ૈ': 'ai', 'ો': 'o', 'ૌ': 'au',
}

_GUJARATI_CONSONANTS = {
    'ક': 'ka', 'ખ': 'kha', 'ગ': 'ga', 'ઘ': 'gha', 'ઙ': 'nga',
    'ચ': 'cha', 'છ': 'chha', 'જ': 'ja', 'ઝ': 'jha', 'ઞ': 'nya',
    'ટ': 'ta', 'ઠ': 'tha', 'ડ': 'da', 'ઢ': 'dha', 'ણ': 'na',
    'ત': 'ta', 'થ': 'tha', 'દ': 'da', 'ધ': 'dha', 'ન': 'na',
    'પ': 'pa', 'ફ': 'pha', 'બ': 'ba', 'ભ': 'bha', 'મ': 'ma',
    'ય': 'ya', 'ર': 'ra', 'લ': 'la', 'વ': 'va', 'શ': 'sha',
    'ષ': 'sha', 'સ': 'sa', 'હ': 'ha', 'ળ': 'la',
    'ક્ષ': 'ksha', 'જ્ઞ': 'gya', 'ત્ર': 'tra',
}

_GUJARATI_SPECIALS = {
    'ં': 'n', 'ઃ': 'h', 'ઁ': 'n',
    '૦': '0', '૧': '1', '૨': '2', '૩': '3', '૪': '4',
    '૫': '5', '૬': '6', '૭': '7', '૮': '8', '૯': '9',
    '।': '. ', '॥': ' || ',
}

_GUJARATI_VIRAMA = '્'


def _transliterate_gujarati(text: str) -> str:
    return _transliterate_brahmic(text, _GUJARATI_CONSONANTS, _GUJARATI_VOWELS,
                                  _GUJARATI_MATRAS, _GUJARATI_SPECIALS, _GUJARATI_VIRAMA)


# ---------------------------------------------------------------------------
# Odia (U+0B00–U+0B7F)
# ---------------------------------------------------------------------------

_ODIA_VOWELS = {
    'ଅ': 'a', 'ଆ': 'aa', 'ଇ': 'i', 'ଈ': 'ii', 'ଉ': 'u', 'ଊ': 'uu',
    'ଋ': 'ri', 'ଏ': 'e', 'ଐ': 'ai', 'ଓ': 'o', 'ଔ': 'au',
}

_ODIA_MATRAS = {
    'ା': 'aa', 'ି': 'i', 'ୀ': 'ii', 'ୁ': 'u', 'ୂ': 'uu', 'ୃ': 'ri',
    'େ': 'e', 'ୈ': 'ai', 'ୋ': 'o', 'ୌ': 'au',
}

_ODIA_CONSONANTS = {
    'କ': 'ka', 'ଖ': 'kha', 'ଗ': 'ga', 'ଘ': 'gha', 'ଙ': 'nga',
    'ଚ': 'cha', 'ଛ': 'chha', 'ଜ': 'ja', 'ଝ': 'jha', 'ଞ': 'nya',
    'ଟ': 'ta', 'ଠ': 'tha', 'ଡ': 'da', 'ଢ': 'dha', 'ଣ': 'na',
    'ତ': 'ta', 'ଥ': 'tha', 'ଦ': 'da', 'ଧ': 'dha', 'ନ': 'na',
    'ପ': 'pa', 'ଫ': 'pha', 'ବ': 'ba', 'ଭ': 'bha', 'ମ': 'ma',
    'ଯ': 'ya', 'ର': 'ra', 'ଲ': 'la', 'ଳ': 'la', 'ଵ': 'va',
    'ଶ': 'sha', 'ଷ': 'sha', 'ସ': 'sa', 'ହ': 'ha',
    'କ୍ଷ': 'ksha', 'ତ୍ର': 'tra', 'ଜ୍ଞ': 'gya',
}

_ODIA_SPECIALS = {
    'ଂ': 'n', 'ଃ': 'h', 'ଁ': 'n',
    '୦': '0', '୧': '1', '୨': '2', '୩': '3', '୪': '4',
    '୫': '5', '୬': '6', '୭': '7', '୮': '8', '୯': '9',
    '।': '. ', '॥': ' || ',
}

_ODIA_VIRAMA = '୍'


def _transliterate_odia(text: str) -> str:
    return _transliterate_brahmic(text, _ODIA_CONSONANTS, _ODIA_VOWELS,
                                  _ODIA_MATRAS, _ODIA_SPECIALS, _ODIA_VIRAMA)


# ---------------------------------------------------------------------------
# Sinhala (U+0D80–U+0DFF)
# ---------------------------------------------------------------------------

_SINHALA_VOWELS = {
    'අ': 'a', 'ආ': 'aa', 'ඇ': 'ae', 'ඈ': 'aae', 'ඉ': 'i', 'ඊ': 'ii',
    'උ': 'u', 'ඌ': 'uu', 'ඍ': 'ru', 'එ': 'e', 'ඒ': 'ee', 'ඓ': 'ai',
    'ඔ': 'o', 'ඕ': 'oo', 'ඖ': 'au',
}

_SINHALA_MATRAS = {
    'ා': 'aa', 'ැ': 'ae', 'ෑ': 'aae', 'ි': 'i', 'ී': 'ii', 'ු': 'u',
    'ූ': 'uu', 'ෘ': 'ru', 'ෙ': 'e', 'ේ': 'ee', 'ෛ': 'ai',
    'ො': 'o', 'ෝ': 'oo', 'ෞ': 'au',
}

_SINHALA_CONSONANTS = {
    'ක': 'ka', 'ඛ': 'kha', 'ග': 'ga', 'ඝ': 'gha', 'ඞ': 'nga',
    'ච': 'cha', 'ඡ': 'chha', 'ජ': 'ja', 'ඣ': 'jha', 'ඤ': 'nya',
    'ඥ': 'gnya', 'ට': 'ta', 'ඨ': 'tha', 'ඩ': 'da', 'ඪ': 'dha',
    'ණ': 'na', 'ත': 'ta', 'ථ': 'tha', 'ද': 'da', 'ධ': 'dha',
    'න': 'na', 'ප': 'pa', 'ඵ': 'pha', 'බ': 'ba', 'භ': 'bha',
    'ම': 'ma', 'ය': 'ya', 'ර': 'ra', 'ල': 'la', 'ව': 'va',
    'ශ': 'sha', 'ෂ': 'sha', 'ස': 'sa', 'හ': 'ha', 'ළ': 'la',
    'ෆ': 'fa',
}

_SINHALA_SPECIALS = {
    'ං': 'n', 'ඃ': 'h', 'ඁ': 'n',
    '෦': '0', '෧': '1', '෨': '2', '෩': '3', '෪': '4',
    '෫': '5', '෬': '6', '෭': '7', '෮': '8', '෯': '9',
}

_SINHALA_VIRAMA = '්'


def _transliterate_sinhala(text: str) -> str:
    return _transliterate_brahmic(text, _SINHALA_CONSONANTS, _SINHALA_VOWELS,
                                  _SINHALA_MATRAS, _SINHALA_SPECIALS, _SINHALA_VIRAMA)


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

def _detect_script(text: str) -> str:
    """Detect the dominant non-Latin script in text."""
    counts = {
        'cyrillic': 0, 'devanagari': 0, 'arabic': 0, 'japanese': 0,
        'korean': 0, 'greek': 0, 'hebrew': 0, 'thai': 0, 'latin': 0,
        'telugu': 0, 'tamil': 0, 'malayalam': 0, 'kannada': 0,
        'bengali': 0, 'gurmukhi': 0, 'gujarati': 0, 'odia': 0,
        'sinhala': 0, 'chinese': 0,
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
        elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            counts['chinese'] += 1
        elif 0xAC00 <= code <= 0xD7A3:
            counts['korean'] += 1
        elif 0x0370 <= code <= 0x03FF:
            counts['greek'] += 1
        elif 0x0590 <= code <= 0x05FF:
            counts['hebrew'] += 1
        elif 0x0E00 <= code <= 0x0E7F:
            counts['thai'] += 1
        elif 0x0C00 <= code <= 0x0C7F:
            counts['telugu'] += 1
        elif 0x0B80 <= code <= 0x0BFF:
            counts['tamil'] += 1
        elif 0x0D00 <= code <= 0x0D7F:
            counts['malayalam'] += 1
        elif 0x0C80 <= code <= 0x0CFF:
            counts['kannada'] += 1
        elif 0x0980 <= code <= 0x09FF:
            counts['bengali'] += 1
        elif 0x0A00 <= code <= 0x0A7F:
            counts['gurmukhi'] += 1
        elif 0x0A80 <= code <= 0x0AFF:
            counts['gujarati'] += 1
        elif 0x0B00 <= code <= 0x0B7F:
            counts['odia'] += 1
        elif 0x0D80 <= code <= 0x0DFF:
            counts['sinhala'] += 1
        elif ch.isalpha():
            counts['latin'] += 1

    non_latin = {k: v for k, v in counts.items() if k != 'latin'}
    if not non_latin or max(non_latin.values()) == 0:
        return 'latin'
    return max(non_latin, key=non_latin.get)


# Human-readable language / region for each detected script. Scripts shared by
# several languages (Devanagari, Arabic, Bengali, …) list the main ones.
_SCRIPT_LANGUAGES = {
    'latin': 'Latin script (English / European / romanized)',
    'cyrillic': 'Cyrillic (Russian, Ukrainian, Bulgarian, …)',
    'devanagari': 'Devanagari (Hindi, Marathi, Nepali, Sanskrit)',
    'arabic': 'Arabic script (Arabic, Urdu, Persian)',
    'japanese': 'Japanese (Hiragana / Katakana)',
    'chinese': 'Chinese (Han)',
    'korean': 'Korean (Hangul)',
    'greek': 'Greek',
    'hebrew': 'Hebrew',
    'thai': 'Thai',
    'telugu': 'Telugu',
    'tamil': 'Tamil',
    'malayalam': 'Malayalam',
    'kannada': 'Kannada',
    'bengali': 'Bengali / Assamese',
    'gurmukhi': 'Punjabi (Gurmukhi)',
    'gujarati': 'Gujarati',
    'odia': 'Odia (Oriya)',
    'sinhala': 'Sinhala',
}

# Indic scripts specifically — surfaced as "Indic language detected".
INDIC_SCRIPTS = {
    'devanagari', 'bengali', 'gurmukhi', 'gujarati', 'odia',
    'tamil', 'telugu', 'kannada', 'malayalam', 'sinhala',
}


def detect_language(text: str) -> Dict[str, object]:
    """Detect the script/language of lyrics text.

    Returns a dict with the dominant ``script``, a human-readable
    ``language`` name, whether it is an ``indic`` script, whether it
    ``needs_transliteration``, and the per-script character counts.
    """
    counts: Dict[str, int] = {}
    for line in (text or "").split('\n'):
        script = _detect_script(line)
        counts[script] = counts.get(script, 0) + sum(1 for c in line if c.isalpha())
    script = _detect_script(text or "")
    return {
        "script": script,
        "language": _SCRIPT_LANGUAGES.get(script, script),
        "indic": script in INDIC_SCRIPTS,
        "needs_transliteration": bool(text) and script != 'latin',
        "counts": counts,
    }


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
    'telugu': _transliterate_telugu,
    'tamil': _transliterate_tamil,
    'malayalam': _transliterate_malayalam,
    'kannada': _transliterate_kannada,
    'bengali': _transliterate_bengali,
    'gurmukhi': _transliterate_gurmukhi,
    'gujarati': _transliterate_gujarati,
    'odia': _transliterate_odia,
    'sinhala': _transliterate_sinhala,
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
