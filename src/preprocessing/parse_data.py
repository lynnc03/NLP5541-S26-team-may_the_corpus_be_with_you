#parse_data.py

"""
Purpose: parses a talkbank .cha data file into a ParsedSpeech object

input: .cha file

returns: parsed data file

How to use: 
    from parse_data.py import ParsedSpeech
    parser = ParsedSpeech()
    speech_parsed = parser.parse("[insert_path_to_file].chat", row_in_master_file)

Steps: parse header
    extract speech
    extract CHAT features
    each type of % collected
    #find age/sex
    #context windows
"""
import sys
import re
import pylangacq       
from typing import Optional
import json
from pathlib import Path
from data_classes import (ParsedSpeech, Metadata, Utterance, CHATFeatures, Annotations)

class CHAFileParser:
    def __init__(self, prior_context: int = 2, later_context: int = 2):
        self.prior_context = prior_context
        self.later_context = later_context


    #let's just pause to say that I hate regex. I wish there was a better option than regex. But, alas.
    speech_lines_dict = {
        "speech_line": re.compile(r'^\*([A-Z][A-Z0-9]*):[ \t]*(.*)'),
        "annotation_line": re.compile(r'^%([a-z]+):[ \t]*(.*)'),
        "id_line": re.compile(r'^@ID:[ \t]*(.*)'),
        "age": re.compile(r'^(\d+);(\d+)(?:\.(\d+))?$'),
        "age_year": re.compile(r'^(\d+)$'),
        #could add parsing here later for sound
    }

    chat_notation = {
        "long_pause": re.compile(r'\(\.\.\.\)'),
        "med_pause": re.compile(r'\(\.\.\)'),
        "short_pause": re.compile(r'\(\.\)'),
        "pause_length": re.compile(r'\((\d+\.?\d*)\)'),
        "filled": re.compile(r'&(?:uh|um|er|ah|hm)\b', re.IGNORECASE),
        "repeat": re.compile(r'\[/\]'),
        "revision": re.compile(r'\[//\]'),
        "reformulation": re.compile(r'\[///\]'),
        "trailing": re.compile(r'\+\.\.\.'),
        "interrupted": re.compile(r'\+/\.'),
        "error": re.compile(r'\[\*\]'),
        "target": re.compile(r'\[:\s*\w+'),
        "uncertain": re.compile(r'\[\?\]'),
        "xxx_unintelligible": re.compile(r'\bxxx\b'),
        "yyy_phonological": re.compile(r'\byyy\b'),
        "paralinguistic": re.compile(r'&=\w+'),
        "timestamp": re.compile(r'\x15\d+_\d+\x15')
    }

    def parse(self, file_path: str, master_row: dict) -> ParsedSpeech:
        """
        returns ParsedSpeech with metadata, utterances, features, and all other info
        """
        filepath = Path(file_path)
        if filepath.exists():
            lines = self._read_file(filepath)

            #parse the header
            header, main_lines = self._split_header_main(lines)

            header_data = self._parse_header(header)
            check_header = self._check_master(header_data, master_row)
            header_metadata = self._build_metadata(header_data, master_row, check_header)


            #speech
            raw_speech = self._parse_utterances(filepath, master_row["file_id"])
            speech = [self._build_speech(speech_line) for speech_line in raw_speech]
            speech = self._attach_context_windows(speech)

            parsed_speech = ParsedSpeech(metadata= header_metadata, utterances = speech)

            return parsed_speech 
        
        else:
            raise FileNotFoundError(f"File not found: {filepath}")


    def _read_file(self, filepath: Path) -> list[str]:
        "reads utf-8"
        try:
            try:
                lines = filepath.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                lines = filepath.read_text(encoding="latin-1").splitlines()

            return lines
        except Exception as e:
            print(f"File error Exception: {e}")
        
    def _split_header_main(self, lines: list[str]) -> tuple[list[str], list[str]]:
        split = len(lines)
        for idx, line in enumerate(lines):
            if line.startswith("*"):
                split = idx
                break #breaks if not
                
        header, main = lines[:split], lines[split:]
    
        return header, main

    def _parse_utterances(self, filepath: Path, file_id: str) -> list[dict]:
        """
        Use pylangacq to extract utterances and dependent tiers.
        Returns list of raw dicts matching the structure _build_speech() expects.
        """
        chat = pylangacq.read_chat(str(filepath), strict=False)
        
        raw_speech = []
        speech_idx = 0

        for utt in chat.utterances():
            speech_idx += 1

            speaker  = utt.participant or "UNKNOWN"
            tiers = utt.tiers or {}
            text_raw = tiers.get(speaker) or ""

            if not text_raw.strip(): 
                continue 
            
            annotations = {
                k: v for k, v in tiers.items() if k != speaker}

            raw_speech.append({
                "file_id": file_id,
                "utterance_index": speech_idx,
                "speaker": speaker,
                "text_raw": text_raw,
                "annotations": annotations,
                "parse_warnings": [],})

        return raw_speech
    
    def _build_speech(self, speech_line: dict) -> Utterance:
        features = self._extract_features(speech_line["text_raw"])
        annotations = self._extract_annotations(speech_line["annotations"])

        utterance = Utterance(file_id = speech_line.get("file_id", ""),
                  utterance_index = speech_line["utterance_index"],
                  speaker = speech_line["speaker"],
                  text_raw = speech_line["text_raw"],
                  features = features,
                  annotations = annotations,
                  parse_warnings = speech_line.get("parse_warnings", []))
        
        return utterance
    
    def _attach_context_windows(self, speech: list[Utterance]) -> list[Utterance]:
        for idx, utt in enumerate(speech):
            if not utt.is_target_child:
                continue

            utt.context_before = [
                {"speaker": speech[j].speaker, "text_raw": speech[j].text_raw}
                for j in range(max(0, idx - self.prior_context), idx)]
            
            utt.context_after = [
                {"speaker": speech[j].speaker, "text_raw": speech[j].text_raw}
                for j in range(idx + 1, min(len(speech), idx + 1 + self.later_context))]
        return speech

    def _parse_header(self, lines: list[str]) -> dict:
        raw_info = {}
        id_info = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("@ID"):
                edited = line.partition(":")[2].strip()
                id_info.append(edited)
            elif line.startswith("@") and ":" in line:
                key, divider, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                raw_info[key] = value

        #id info
        id_records = []
        for id in id_info:
            parsed_id = self._parse_id_line(id)
            if parsed_id:
                id_records.append(parsed_id)

        #participant info
        speaker = self._parse_participants(raw_info.get("@Participants", ""))
        speaker_codes = [sp["code"] for sp in speaker]
    
        examiner_codes = [examiner for examiner in speaker_codes if examiner in Metadata.EXAMINER_CODES]
        parent_codes = [parent for parent in speaker_codes if parent in Metadata.PARENT_CODES]
        multiple_kids = sum(1 for child in speaker_codes if child.startswith("CHI")) > 1

        languages = [language.strip() for language in raw_info.get("@Languages", "").split(",") if language.strip()]        
        types = [t.strip() for t in raw_info.get("@Types", "").split(",") if t.strip()] 
    
        #dict
        header_info = {"pid": raw_info.get("@PID"),
                       "languages": languages, #don't really need this rn
                       "types": types,
                       "participants": speaker,
                       "speaker_codes": speaker_codes,
                       "examiner_codes": examiner_codes,
                       "parent_codes": parent_codes,
                       "multiple_kids": multiple_kids,
                       "id_records": id_records
        }
    
        return header_info
    

    def _check_master(self, header_data: dict, master_row: dict) -> list[str]:
        #don't really need to do this, but want to make sure dat ais consistent based on manual entries, if any, and the cha metadata
        warnings = []
        file_data = None
        for record in header_data["id_records"]:
            if record.get("role") == "Target_Child":
                file_data = record
                break
        
        if file_data is None:
            return ["no target_child ID line in file header"]

        master_sex = str(master_row.get("sex", "") or "").lower()
        metadata_sex = str(file_data.get("sex", "") or "").lower()
        if master_sex and metadata_sex and master_sex != metadata_sex:
            warnings.append(f"sex doesn't match: master file {master_sex} and metadata_sex {metadata_sex}")


        master_age = str(master_row.get("age", "") or "")
        file_age = str(file_data.get("age","") or "")
        if master_age and file_age and master_age != file_age:
            warnings.append(f"age doesn't match - master age is {master_age} and file_age is {file_age}")
        
        return warnings
    
    def _build_metadata(self, header_data: dict, master_row: dict, check_header: list[str]) -> Metadata:
        """metadata object"""

        metadata = Metadata(
            file_id = master_row["file_id"],
            corpus = None,
            label_binary = int(master_row.get("label_binary", -1)),
            label = str(master_row.get("label", "")),
            age = master_row.get("age"),
            sex = master_row.get("sex"),
            session_types = header_data["types"],
            pid = header_data["pid"],
            languages = header_data["languages"],
            speaker_codes = header_data["speaker_codes"],
            examiner_codes = header_data["examiner_codes"],
            parent_codes = header_data["parent_codes"],
            multiple_kids = header_data["multiple_kids"],
            header_warnings = check_header,
        )

        return metadata

    def _parse_id_line(self, id: str) -> Optional[dict]:
        fields = id.strip().split("|")
        #too short
        if len(fields)<8:
            return None
        
        #is it always here?
        age = fields[3].strip()
        age_months, age_error = self._parse_age(age)

        id_info = {"language": fields[0].strip(),
                   "corpus": fields[1].strip(),
                   "speaker": fields[2].strip(),
                   "age": age,
                   "age_months": age_months,
                   "age_error": age_error,
                   "sex": fields[4].strip() or None,
                   "role": fields[7].strip()}

        return id_info

    def _parse_participants(self, participant_str: str) -> list[dict]:
        #parwse participant value into {code, name, role}
        if not participant_str:
            return []
        
        participant_info = []
        for info in participant_str.split(","):
            parts = info.strip().split()
            if parts:
                participant_info.append({
                    "code": parts[0],
                    "name": parts[1] if len(parts) > 1 else None,
                    "role": parts[2] if len(parts) > 2 else None,
                })

        return participant_info

    def _parse_age(self, age_str: str) -> tuple[Optional[int], Optional[str]]:
        #not everything had age info attached
        if not age_str:
            return None, "no age info"
        
        age_str = age_str.rstrip(".")

        #format could be 3;05 or 3;05.12
        info = self.speech_lines_dict["age"].match(age_str)
        if info:
            years = int(info.group(1))
            months = int(info.group(2))
            total_months = years*12 + months
            return total_months, None
        
        info_year = self.speech_lines_dict["age_year"].match(age_str)
        if info_year:
            years = int(info_year.group(1))
            return years*12, None
        
        return None, f"no age available: {age_str}"

    def _extract_features(self, text: str) -> CHATFeatures:

        if not text or not text.strip():
            return CHATFeatures()
        
        ch_no = self.chat_notation
        text = ch_no["timestamp"].sub("", text)

        pause_long   = len(ch_no["long_pause"].findall(text))
        pause_medium = len(ch_no["med_pause"].findall(text))
        pause_short  = len(ch_no["short_pause"].findall(text))

        #strpping pauses so no overocunting
        t = ch_no["long_pause"].sub("", text)
        t = ch_no["med_pause"].sub("", t)
        t = ch_no["short_pause"].sub("", t)

        timed_vals = ch_no["pause_length"].findall(t)
        pause_timed_count = len(timed_vals)
        pause_timed_sec = round(sum(float(v) for v in timed_vals), 3)

        filled    = len(ch_no["filled"].findall(text))
        repeat    = len(ch_no["repeat"].findall(text))
        revision  = len(ch_no["revision"].findall(text))
        reform    = len(ch_no["reformulation"].findall(text))
        trailing  = len(ch_no["trailing"].findall(text))
        interrupt = len(ch_no["interrupted"].findall(text))
        errors    = len(ch_no["error"].findall(text))
        targets   = len(ch_no["target"].findall(text))
        uncertain = len(ch_no["uncertain"].findall(text))
        xxx       = len(ch_no["xxx_unintelligible"].findall(text))
        yyy       = len(ch_no["yyy_phonological"].findall(text))
        para      = len(ch_no["paralinguistic"].findall(text))

        chat_features = CHATFeatures(
            pause_short = pause_short, pause_medium = pause_medium, pause_long = pause_long,
            pause_timed_count = pause_timed_count, pause_timed_total_sec = pause_timed_sec,
            pause_total = pause_short + pause_medium + pause_long + pause_timed_count,
            filled_pause_count = filled, repetition_count = repeat,
            revision_count = revision,
            reformulation_count = reform,
            trailing_off_count = trailing,
            interrupted_count = interrupt,
            disfluency_total = filled + repeat + revision + reform,
            error_count = errors,
            target_form_count = targets,
            uncertain_count = uncertain,
            unintelligible_xxx = xxx, unintelligible_yyy = yyy,
            unintelligible_total = xxx + yyy, paralinguistic_count = para,)
        
        return chat_features
    
    def _extract_annotations(self, annotations: dict) -> Annotations:
        mor_raw = annotations.get("%mor")
        mor_count = None
        if mor_raw:
            tokens = [token for token in mor_raw.split() if "|" in token]
            mor_count = len(tokens)

        err_raw   = annotations.get("%err")
        err_count = 0
        if err_raw:
            err_count = len([
                token for token in err_raw.split() if token not in (".", "!", "?")])

        tim_raw    = annotations.get("%tim")
        tim_onset  = None
        tim_offset = None
        tim_dur    = None
        if tim_raw:
            om = re.search(r'onset[=:\s]+(\d+)', tim_raw, re.IGNORECASE)
            fm = re.search(r'offset[=:\s]+(\d+)', tim_raw, re.IGNORECASE)
            if om:
                tim_onset = int(om.group(1))
            if fm:
                tim_offset = int(fm.group(1))
            if tim_onset is not None and tim_offset is not None:
                tim_dur = tim_offset - tim_onset

        annotations_obj = Annotations(                              
            mor_raw = mor_raw,
            mor_token_count = mor_count,
            gra_raw = annotations.get("%gra"),
            tim_raw = tim_raw,
            tim_onset = tim_onset,
            tim_offset = tim_offset,
            tim_duration_ms = tim_dur,
            err_tier_raw = err_raw,
            err_tier_count = err_count,
            com_raw = annotations.get("%com"),
            act_raw = annotations.get("%act"),
            gpx_raw = annotations.get("%gpx"),)

        return annotations_obj

    def _attach_context_windows(self, speech: list[Utterance]) -> list[Utterance]:
        for idx, utt in enumerate(speech):
            if not utt.is_target_child:
                continue

            utt.context_before = [
                {"speaker": speech[j].speaker, "text_raw": speech[j].text_raw}
                for j in range(max(0, idx - self.prior_context), idx)
            ]
            utt.context_after = [
                {"speaker": speech[j].speaker, "text_raw": speech[j].text_raw}
                for j in range(idx + 1, min(len(speech), idx + 1 + self.later_context))
            ]
        return speech
    
if __name__ == "__main__":

    if len(sys.argv) < 2 :
        print("please use command python3 parse_data.py [insert_file_path_here].cha")
        sys.exit(1)

    fake_metadata = {"file_id": Path(sys.argv[1]).stem,
                     "label_binary": 0,
                     "label": "control",
                     "age": None,
                     "sex": None}
    
    parser = CHAFileParser()
    test_parser = parser.parse(sys.argv[1], fake_metadata)

    print(f"file_id: {test_parser.file_id}")
    #what else do we care to print here?

    for speech in test_parser.utterances[:5]:
        print(json.dumps(speech.to_dict(), indent=2, default=str))
