Talkbank_project includes:

##NOTE: this is only for v1. V1 includes only controls and children marked SLI/language_disorders. (Some just said "language_disorder" which was unspecified. This was a small number. We can remove if needed)


##Requirements.txt

	Run pip install -r requirements.txt to install required packages

##Data files:

No sound level data included at this time as only a few datasets have this.

Processed files are located in talkbank_project/data/processed

	1. child_utterances.csv

		About: One row for child utterance. This is the raw text, with cleaned text variance, all CHAT feature counts like pauses, disfuencies, etc. Also includes morphological annotations and metadata like age, sex.
		
		Label indicates if disordered or not.

	2. all_utterances.csv
	
		About: The same as child_utterances but includes the parent, sibling, or other person talking with the child who is the focus. 
	
		Useful if you want conversational context or interactions beyond just how the child talks.

	3. child_context_windows.csv

		About: child utterances, but each row also has the context before the child spoke and after. 

		Useful for conversations, or just including broader context.

	4. session_level.csv
		
		About: One row per speech recording. Features are summed and averaged across the number of child utterences. 

	5. pipeline_warnings.csv

		About: Not actually data. Includes information on cases when the age or sex of a child doesn't match what is in files_master.csv. This could happen due to parsing errors. Isn't a huge number of cases, but these could be 		addressed later on, time permitting
		
Raw files can be found in talkbank_project/data/raw.

##Master File Info:

	1.file_info/files_master.csv

		A csv containing information about all files in the dataset. One row per .cha file. Columns include file_id, file_path, label, label_binary, age, sex, include_v1.
		include_v1 notes if a given file was included in the first version of the pipeline, which included only controls, and children marked "SLI" or "lang_disorder" (small number of files just said "lang disorder")

##Notebooks

	1. notebooks/nlp_project_exploration.ipynb

		Just mesing around inspecting the way the .cha files are set up. Not EDA. Just for basic data processing.

##USEFUL NOTES:

	label_binary = 0 if typically developing control
	1 = SLI or language disorder (other types of language disorders not included for V1 like down syndrome, hearing loss, late_talker, etc.) Will be very easy to include these in future.
