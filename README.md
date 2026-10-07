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
