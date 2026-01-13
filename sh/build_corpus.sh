#!/bin/zsh
#
# build_corpus.sh - Download Standard Ebooks and build a training corpus
#
# Usage: ./build_corpus.sh [output_file]
#
# This script:
# 1. Clones Standard Ebooks repos listed in BOOKS array below
# 2. Extracts plain text from XHTML chapter files
# 3. Combines everything into a single corpus file
#
# Prerequisites:
# - git
# - pandoc (brew install pandoc) - for clean HTML-to-text conversion
#

set -e

# Output file (default or first argument)
OUTPUT_FILE="${1:-corpus_from_standard_ebooks.txt}"

# Temporary directory for cloned repos
WORK_DIR="$(mktemp -d)"
trap "rm -rf $WORK_DIR" EXIT

# ============================================================
# BOOK LIST - Add or remove repos here
# Find repos at: https://github.com/standardebooks
# ============================================================
BOOKS=(
    # Jane Austen
    "jane-austen_pride-and-prejudice"
    "jane-austen_emma"
    "jane-austen_sense-and-sensibility"
    "jane-austen_persuasion"
    "jane-austen_mansfield-park"
    "jane-austen_northanger-abbey"

    # Charles Dickens
    "charles-dickens_great-expectations"
    "charles-dickens_a-tale-of-two-cities"
    "charles-dickens_oliver-twist"
    "charles-dickens_david-copperfield"
    "charles-dickens_bleak-house"
    "charles-dickens_our-mutual-friend"

    # George Eliot
    "george-eliot_middlemarch"
    "george-eliot_the-mill-on-the-floss"
    "george-eliot_silas-marner"
    "george-eliot_daniel-deronda"

    # Thomas Hardy
    "thomas-hardy_tess-of-the-durbervilles"
    "thomas-hardy_jude-the-obscure"
    "thomas-hardy_far-from-the-madding-crowd"
    "thomas-hardy_the-mayor-of-casterbridge"
    "thomas-hardy_the-return-of-the-native"

    # Joseph Conrad
    "joseph-conrad_lord-jim"
    "joseph-conrad_heart-of-darkness"
    "joseph-conrad_nostromo"

    # Henry James
    "henry-james_the-portrait-of-a-lady"
    "henry-james_the-ambassadors"
    "henry-james_washington-square"
)

# ============================================================
# Check prerequisites
# ============================================================
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is required but not installed."
    echo "Install with: brew install pandoc"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "Error: git is required but not installed."
    exit 1
fi

# ============================================================
# Main processing
# ============================================================
echo "========================================"
echo "Standard Ebooks Corpus Builder"
echo "========================================"
echo "Output file: $OUTPUT_FILE"
echo "Working directory: $WORK_DIR"
echo "Books to process: ${#BOOKS[@]}"
echo "========================================"
echo ""

# Clear output file
> "$OUTPUT_FILE"

PROCESSED=0
FAILED=0

for book in "${BOOKS[@]}"; do
    echo "Processing: $book"

    REPO_URL="https://github.com/standardebooks/${book}.git"
    REPO_DIR="$WORK_DIR/$book"

    # Clone (shallow, just the latest)
    if git clone --depth 1 --quiet "$REPO_URL" "$REPO_DIR" 2>/dev/null; then

        # Find the text directory
        TEXT_DIR="$REPO_DIR/src/epub/text"

        if [[ -d "$TEXT_DIR" ]]; then
            # Extract book title for header
            TITLE=$(echo "$book" | sed 's/-/ /g' | sed 's/_/ - /g')
            echo "" >> "$OUTPUT_FILE"
            echo "# $TITLE" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"

            # Process chapter files in order
            # Skip front/back matter, focus on actual content
            # Use (N) glob qualifier to allow empty matches without error
            for xhtml in "$TEXT_DIR"/chapter-*.xhtml(N) "$TEXT_DIR"/epilogue-*.xhtml(N); do
                if [[ -f "$xhtml" ]]; then
                    # Convert XHTML to plain text, append to corpus
                    pandoc -f html -t plain --wrap=none "$xhtml" >> "$OUTPUT_FILE" 2>/dev/null
                    echo "" >> "$OUTPUT_FILE"
                fi
            done

            echo "  Done"
            PROCESSED=$((PROCESSED + 1))
        else
            echo "  Warning: No text directory found"
            FAILED=$((FAILED + 1))
        fi

        # Clean up this repo to save space
        rm -rf "$REPO_DIR"
    else
        echo "  Failed to clone (repo may not exist or name may differ)"
        FAILED=$((FAILED + 1))
    fi
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "Complete!"
echo "========================================"
echo "Processed: $PROCESSED books"
echo "Failed: $FAILED books"
echo ""

if [[ -f "$OUTPUT_FILE" ]]; then
    CHARS=$(wc -c < "$OUTPUT_FILE" | tr -d ' ')
    LINES=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')
    echo "Output: $OUTPUT_FILE"
    echo "Size: $CHARS characters, $LINES lines"
fi
