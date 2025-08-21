
pip install wikiextractor
pip install six
pip install nltk

#!/bin/bash

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Project root: $ROOT"

cd "$ROOT/wikidata" || { echo "Error: Directory $ROOT/wikidata not found"; exit 1; }

WIKI_EXTRACTED_DIR="text"
DUMP_FILE="enwiki-latest-pages-articles1.xml-p1p41242.bz2"
if [ -d "$WIKI_EXTRACTED_DIR" ]; then
    if [ -n "$(ls -A $WIKI_EXTRACTED_DIR 2>/dev/null)" ]; then
        echo "$WIKI_EXTRACTED_DIR/ already exists and is not empty. Skipping WikiExtractor."
    else
        echo "$WIKI_EXTRACTED_DIR/ exists but is empty. Running WikiExtractor..."
        python -m wikiextractor.WikiExtractor --json "$DUMP_FILE"
    fi
else
    echo "🔍 $WIKI_EXTRACTED_DIR/ not found. Running WikiExtractor..."
    python -m wikiextractor.WikiExtractor --json "$DUMP_FILE"
fi


WIKI_EXTRACTED_DIR="text"            # WikiExtractor 输出的文本目录
OUTPUT_DIR="output"                  # 合并和预处理输出目录
MERGED_JSON="$OUTPUT_DIR/wiki_all.json"

mkdir -p "$OUTPUT_DIR"
> "$MERGED_JSON"

echo "Merging JSON files from $WIKI_EXTRACTED_DIR ..."
find "$WIKI_EXTRACTED_DIR" -type f -print0 | \
while IFS= read -r -d '' file; do
    cat "$file" >> "$MERGED_JSON"
done

echo "File merge success: $MERGED_JSON"

MEGATRON_DIR="$ROOT/../thirdparty/nvidia_megatron_lm_0_9_0"
cd "$MEGATRON_DIR" || { echo "Error: Megatron-LM directory not found at $MEGATRON_DIR"; exit 1; }

OUTPUT_PREFIX="$ROOT/wikidata/$OUTPUT_DIR/my-gpt2"
mkdir -p "$(dirname "$OUTPUT_PREFIX")"

VOCAB_FILE="$ROOT/gpt2-vocab.json"
MERGE_FILE="$ROOT/gpt2-merges.txt"

if [ ! -f "$VOCAB_FILE" ]; then
    echo "Error: vocab file not found at $VOCAB_FILE"
    exit 1
fi
if [ ! -f "$MERGE_FILE" ]; then
    echo "Error: merge file not found at $MERGE_FILE"
    exit 1
fi

echo "Starting data preprocessing with Megatron-LM..."
python tools/preprocess_data.py \
    --input "$ROOT/wikidata/$MERGED_JSON" \
    --output-prefix "$OUTPUT_PREFIX" \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --vocab-file "$VOCAB_FILE" \
    --merge-file "$MERGE_FILE" \
    --workers 64

cd - > /dev/null

echo "Preprocessing completed! Output files are in $OUTPUT_DIR/"
