#clean_text.py


import re
from data_classes import ParsedSpeech, Utterance


class TextCleaner:

    TOKENS = {
        "pause": "[PAUSE]",
        "repeat": "[REPEAT]",
        "revision": "[REVISION]",
        "reformulation": "[REFORMULATION]",
        "error": "[ERROR]",
        "unintelligible": "[UNINTELLIGIBLE]",
        "trailing": "[TRAILING]",
        "interrupted": "[INTERRUPTED]",
        "filled_pause": "[FILLED_PAUSE]",
    }

    def __init__(self):
        self._target_form      = re.compile(r'(\S+)\s*\[:\s*([^\]]+)\]')
        self._error_marker = re.compile(r'\[\*\]')
        self._repetition = re.compile(r'\[/\]')
        self._revision = re.compile(r'\[//\]')
        self._reformulation = re.compile(r'\[///\]')
        self._disfluency_any = re.compile(r'\[/{1,3}\]')
        self._scope = re.compile(r'[<>]')
        self._pause_long = re.compile(r'\(\.\.\.\)')
        self._pause_med = re.compile(r'\(\.\.\)')
        self._pause_short = re.compile(r'\(\.\)')
        self._pause_timed = re.compile(r'\(\d+\.?\d*\)')
        self._filled_pause = re.compile(r'&(uh|um|er|ah|hm)\b', re.IGNORECASE)
        self._paralinguistic = re.compile(r'&=\w+')
        self._unintelligible = re.compile(r'\b(xxx|yyy)\b')
        self._trailing_off = re.compile(r'\+\.\.\.')
        self._interrupted = re.compile(r'\+/\.')
        self._uncertain = re.compile(r'\[\?\]')
        self._remaining_brackets = re.compile(r'\[[^\]]*\]')
        self._fragment = re.compile(r'&\w+')
        self._lengthening = re.compile(r'::+')
        self._timestamp = re.compile(r'\x15\d+_\d+\x15')

        self._terminal = re.compile(r'\s*[.!?]\s*$')
        self._multi_space = re.compile(r'  +')

    def apply(self, session: ParsedSpeech) -> None:
        for utt in session.utterances:
                self._apply_to_utterance(utt)

    def clean(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "text_raw": "",
                "text_clean": "",
                "text_surface": "",
                "text_disfluency_tagged": ""}
        return {"text_raw": text,
            "text_clean": self._make_clean(text),
            "text_surface": self._make_surface(text),
            "text_disfluency_tagged": self._make_disfluency_tagged(text)}

    def _apply_to_utterance(self, utt: Utterance) -> None:
        text = utt.text_raw
        if not text or not text.strip():
            utt.text_clean = ""
            utt.text_surface = ""
            utt.text_disfluency_tagged = ""
            return
        utt.text_clean = self._make_clean(text)
        utt.text_surface = self._make_surface(text)
        utt.text_disfluency_tagged = self._make_disfluency_tagged(text)

    def _clean_target_form_repl(self, m):
        return m.group(2).strip()

    def _make_clean(self, text: str) -> str:
        t = text
        t = self._timestamp.sub("", t)

        t = self._target_form.sub(self._clean_target_form_repl, t)
        t = self._error_marker.sub("", t)
        t = self._disfluency_any.sub("", t)
        t = self._scope.sub("", t)
        t = self._pause_long.sub("", t)
        t = self._pause_med.sub("", t)
        t = self._pause_short.sub("", t)
        t = self._pause_timed.sub("", t)
        t = self._filled_pause.sub("", t)
        t = self._paralinguistic.sub("", t)
        t = self._unintelligible.sub("", t)
        t = self._trailing_off.sub("", t)
        t = self._interrupted.sub("", t)
        t = self._uncertain.sub("", t)
        t = self._remaining_brackets.sub("", t)
        t = self._fragment.sub("", t)
        t = self._lengthening.sub("", t)
        t = self._terminal.sub("", t)
        return self._normalize(t)
    
    def _surface_target_form_repl(self, m):
        return m.group(1).strip()

    def _disfluency_target_form_repl(self, m):
        tokens = self.TOKENS
        return f"{m.group(2).strip()} {tokens['error']}"

    def _make_surface(self, text: str) -> str:
        
        t = text
        t = self._timestamp.sub("", t)
        t = self._target_form.sub(self._surface_target_form_repl, t)
        t = self._error_marker.sub("", t)
        t = self._disfluency_any.sub("", t)
        t = self._scope.sub("", t)
        t = self._pause_long.sub("", t)
        t = self._pause_med.sub("", t)
        t = self._pause_short.sub("", t)
        t = self._pause_timed.sub("", t)
        t = self._filled_pause.sub("", t)
        t = self._paralinguistic.sub("", t)
        t = self._unintelligible.sub("", t)
        t = self._trailing_off.sub("", t)
        t = self._interrupted.sub("", t)
        t = self._uncertain.sub("", t)
        t = self._remaining_brackets.sub("", t)
        t = self._fragment.sub("", t)
        t = self._lengthening.sub("", t)
        t = self._terminal.sub("", t)
        return self._normalize(t)

    def _make_disfluency_tagged(self, text: str) -> str:
        tokens = self.TOKENS
        t = text
        t = self._timestamp.sub("", t)

        t = self._target_form.sub(self._disfluency_target_form_repl, t)
        t = self._error_marker.sub(tokens["error"], t)
        t = self._reformulation.sub(tokens["reformulation"], t)
        t = self._revision.sub(tokens["revision"], t)
        t = self._repetition.sub(tokens["repeat"], t)
        t = self._scope.sub("", t)
        t = self._pause_long.sub(tokens["pause"], t)
        t = self._pause_med.sub(tokens["pause"], t)
        t = self._pause_short.sub(tokens["pause"], t)
        t = self._pause_timed.sub(tokens["pause"], t)
        t = self._filled_pause.sub(tokens["filled_pause"], t)
        t = self._paralinguistic.sub("", t)
        t = self._unintelligible.sub(tokens["unintelligible"], t)
        t = self._trailing_off.sub(tokens["trailing"], t)
        t = self._interrupted.sub(tokens["interrupted"], t)
        t = self._uncertain.sub("", t)
        t = self._remaining_brackets.sub("", t)
        t = self._fragment.sub("", t)
        t = self._lengthening.sub("", t)
        t = self._terminal.sub("", t)

        normalized = self._normalize(t)

        return normalized

    def _normalize(self, text: str) -> str:
        text = self._multi_space.sub(" ", text).strip()
        return text
