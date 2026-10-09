# BookNLP

A natural language processing pipeline for analyzing works of fiction, including entity detection, quotation attribution, and character relationship analysis.

## Prerequisites

both python & dependencies are managed by `uv` here is [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/akupiec/booknlp
cd booknlp
```

2. Install the required packages:
```bash
uv sync
```

## Testing

Run the complete test suite with:

```bash
uv run pytest
```

## Usage

```bash
uv run booknlp examples/pride.short.txt
```

### Command Line Arguments

- `input_file`: The text file to process (required)
- `--output-dir`: Directory where output files will be saved (default: 'output')
- `--model`: Model size to use - 'big' or 'small' (default: 'small')

### Output Files

The pipeline generates several output files in the specified output directory:

1. `{book_id}.tokens`: Word-level information including:
   - Paragraph and sentence IDs
   - Word forms and lemmas
   - Part-of-speech tags
   - Dependency relations
   - Event annotations

2. `{book_id}.entities`: Named entity information including:
   - Entity types
   - Coreference IDs
   - Text spans

3. `{book_id}.quotes`: Quotation information including:
   - Quoted text
   - Speaker attribution
   - Coreference information

4. `{book_id}.supersense`: Semantic categories for words:
   - Verb categories
   - Noun categories

5. `{book_id}.event`: Event annotations including:
   - Event types
   - Participants
   - Temporal information

6. `{book_id}.book`: JSON file containing:
   - Character information
   - Relationships
   - Actions
   - Attributes

7. `{book_id}.book.html`: Interactive HTML visualization of the text with:
   - Entity annotations
   - Character relationships
   - Interactive features

8. `{book_id}.stage_script.txt`: Plain-text stage-script rendering with:
   - Source text in reading order
   - Unmarked text assigned to `[NARRATOR]`
   - Attributed quotes labeled as `[Name #coreference_id]`
   - System quotes rendered under `[SYSTEM]` without square brackets in the
     spoken text
   - Confidently inferred binary referential gender appended as `M` or `F`
     only when confidence is above 60%
   - Unattributed quotes labeled as `[UNKNOWN]`
