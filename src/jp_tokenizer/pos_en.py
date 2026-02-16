from __future__ import annotations
from typing import Dict, List

# english translation mapping for unidic POS components
# missing components covered by UNK<JP> fallback
POS_COMPONENT_EN: Dict[str, str] = {
    #major POS/POS1
    "名詞": "Noun",
    "動詞": "Verb",
    "形容詞": "Adjective",
    "形状詞": "Adjectival Noun",
    "副詞": "Adverb",
    "連体詞": "Adnominal",
    "接続詞": "Conjunction",
    "感動詞": "Interjection",
    "助詞": "Particle",
    "助動詞": "Auxiliary Verb",
    "代名詞": "Pronoun",
    "記号": "Symbol",
    "補助記号": "Supplementary Symbol",
    "連語": "Compound",
    "接頭辞": "Prefix",
    "接尾辞": "Suffix",
    "その他": "Other",
    "UNK": "Unknown",
    # noun subcats/attributes
    "普通名詞": "Common Noun",
    "固有名詞": "Proper Noun",
    "数詞": "Numeral",
    "人名": "Person Name",
    "姓": "Surname",
    "名": "Given Name",
    "地名": "Place Name",
    "国": "Country",
    "組織": "Organization",
    "一般": "General",
    "副詞可能": "Adverbial Possible",
    "形状詞可能": "Adjectival Noun Possible",
    "サ変可能": "Suru-Verb Possible",
    "サ変接続": "Suru-Verb Compound",
    "非自立可能": "Non-Independent",
    "自立": "Independent",
    "名詞的": "Nominal",
    "数": "Number",
    "引用文字列": "Quoted String",
    # verbs/adjectives
    "非自立": "Non-Independent",
    "可能": "Potential",
    # particles
    "格助詞": "Case Particle",
    "副助詞": "Adverbial Particle",
    "係助詞": "Binding Particle",
    "接続助詞": "Conjunctive Particle",
    "終助詞": "Sentence-Final Particle",
    "準体助詞": "Nominalizing Particle",
    "並立助詞": "Parallel Particle",
    "間投助詞": "Interjectory Particle",
    # symbols/punctuation
    "句点": "Period",
    "読点": "Comma",
    "空白": "Whitespace",
    "括弧開": "Open Bracket",
    "括弧閉": "Close Bracket",
    "補助記号-句点": "Period (Supplementary Symbol)",
    "補助記号-読点": "Comma (Supplementary Symbol)",
    # misc POS2/3
    "一般的": "General",
    "助動詞語幹": "Auxiliary Verb Stem",
    "形容詞語幹": "Adjective Stem",
    "形状詞語幹": "Adjectival Noun Stem",
    "語幹": "Stem",
    "タリ": "Tari (Adjectival)",
}

def translate_pos_components(pos: str) -> List[str]:
    if not pos:
        return []
    parts = [p for p in pos.split("-") if p]
    out: List[str] = []
    for p in parts:
        out.append(POS_COMPONENT_EN.get(p, f"Unknown({p})"))
    return out

def translate_pos(pos: str, sep: str = " / ") -> str:
    comps = translate_pos_components(pos)
    if not comps:
        return ""
    return sep.join(comps)
