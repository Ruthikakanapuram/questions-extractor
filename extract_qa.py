# pdf_to_questions_pipeline.py
#
# Full flow:
# PDF input
#   ↓
# Marker extracts Markdown + images
#   ↓
# Markdown is split into chunks
#   ↓
# Images are copied into output/images
#   ↓
# chunks.json + chunks.md are saved
#   ↓
# OpenAI extracts existing textbook questions from chunks
#   ↓
# questions.json + questions_flat.json + questions.md are saved
#
# Requirements:
# pip install openai python-dotenv
# marker_single should already be installed and available in your environment.
#
# .env:
# OPENAI_API_KEY=your_key_here

import re
import json
import base64
import time
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


# ─────────────────────────────────────────────
# Fixed paths - edit these only
# ─────────────────────────────────────────────

INPUT_PDF = Path("input/5th_maths.pdf")

BASE_OUTPUT_DIR = Path("output")
MARKER_OUTPUT_DIR = BASE_OUTPUT_DIR / "marker_output"
CHUNKS_OUTPUT_DIR = BASE_OUTPUT_DIR / "chunks"
QUESTIONS_OUTPUT_DIR = BASE_OUTPUT_DIR / "questions"

MODEL = "gpt-4o-mini"
DELAY_SECONDS = 1.0


# ─────────────────────────────────────────────
# Prompt for question extraction
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a precise text extraction tool for NCERT/CBSE textbook content.

Your ONLY job is to locate and copy question text VERBATIM from the provided content.

## STRICT EXTRACTION RULES

VERBATIM ONLY:
- Copy question text EXACTLY as it appears.
- Do NOT rephrase, reword, summarize, or clean up any text.
- Do NOT fix grammar, spelling, or punctuation.
- Do NOT translate or transliterate.
- Preserve original numbering, symbols, and formatting.

EXTRACT these question types:
- Numbered problems: 1., 2., Q1., Q.1, Question 1, etc.
- Fill in the blanks
- Match the following
- True / False
- Observe and answer
- Activity prompts: "Try this", "Find out", "Discuss", "Do this"
- Puzzles or games with a clear task
- Sub-parts: (a), (b), (i), (ii)
- Questions that depend on an image, figure, table, or diagram

SKIP entirely:
- Definitions, explanations, or body text
- Solved/worked examples
- "Note for Teachers" or similar teacher-facing content
- Table headers, column labels, or figure captions with no task
- Any text that does not ask the student to do or answer something
- Questions that are incomplete or make no sense without surrounding context

Return ONLY valid JSON. No markdown fences. No commentary. Nothing outside the JSON object.

Schema:
{
  "questions": [
    {
      "question_number": "1",
      "question_text": "<exact text from source>",
      "sub_parts": ["(a) <exact text>", "(b) <exact text>"],
      "has_image": false,
      "question_type": "fill_in_blank | mcq | short_answer | activity | puzzle | arrange | true_false | match | observe_and_answer | other"
    }
  ]
}

If no valid questions are found, return:
{
  "questions": []
}
"""


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class ImageRef:
    alt_text: str
    original_path: str
    source_path: str
    copied_path: str
    markdown_path: str


@dataclass
class Chunk:
    label: str
    chunk_type: str   # exercise | example | section | full_chapter
    exercise_num: str
    content: str
    chapter: Optional[str] = None
    images: list[ImageRef] = field(default_factory=list)


# ─────────────────────────────────────────────
# Step 1: Run Marker
# ─────────────────────────────────────────────

def run_marker_if_needed(input_pdf: Path, marker_output_dir: Path) -> Path:
    """
    Runs Marker on the input PDF and returns the generated Markdown path.
    If Markdown already exists, it reuses the largest .md file.
    """
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    marker_output_dir.mkdir(parents=True, exist_ok=True)

    existing_md_files = list(marker_output_dir.rglob("*.md"))
    if existing_md_files:
        md_path = max(existing_md_files, key=lambda p: p.stat().st_size)
        print(f"Marker Markdown already exists. Reusing: {md_path}")
        return md_path

    command = [
        "marker_single",
        str(input_pdf),
        "--output_dir",
        str(marker_output_dir),
        "--output_format",
        "markdown",
    ]

    print("Running Marker:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        print("Marker STDOUT:")
        print(result.stdout)
        print("Marker STDERR:")
        print(result.stderr)
        raise RuntimeError("Marker failed. Check the error above.")

    md_files = list(marker_output_dir.rglob("*.md"))

    if not md_files:
        raise RuntimeError(f"No Markdown file generated inside: {marker_output_dir}")

    md_path = max(md_files, key=lambda p: p.stat().st_size)

    print(f"Marker Markdown generated: {md_path}")

    return md_path


# ─────────────────────────────────────────────
# Step 2: Markdown chunk extraction
# ─────────────────────────────────────────────

def extract_chapter(text: str) -> Optional[str]:
    match = re.search(
        r'^#{1,2}\s*(Chapter\s+\d+[^\n]*|CHAPTER\s+\d+[^\n]*)',
        text,
        re.MULTILINE | re.IGNORECASE
    )
    return match.group(1).strip() if match else None


def resolve_image_source_path(image_path: str, md_file_path: Path) -> Path:
    image_path = image_path.strip().strip('"').strip("'")

    if image_path.startswith(("http://", "https://")):
        return Path(image_path)

    path = Path(image_path)

    if path.is_absolute():
        return path

    return (md_file_path.parent / path).resolve()


def copy_images_and_rewrite_content(
    content: str,
    md_file_path: Path,
    output_dir: Path,
    chunk_index: int
) -> tuple[str, list[ImageRef]]:
    """
    Finds Markdown image links, copies local images into output/images,
    and rewrites image links so they work from chunks.md.
    """
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)

    image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    images: list[ImageRef] = []

    def replace_image(match: re.Match) -> str:
        alt_text = match.group(1).strip()
        original_path = match.group(2).strip().strip('"').strip("'")

        if original_path.startswith(("http://", "https://")):
            images.append(
                ImageRef(
                    alt_text=alt_text,
                    original_path=original_path,
                    source_path=original_path,
                    copied_path=original_path,
                    markdown_path=original_path,
                )
            )
            return f"![{alt_text}]({original_path})"

        source_path = resolve_image_source_path(original_path, md_file_path)

        if not source_path.exists():
            print(f"⚠ Image not found: {source_path}")
            images.append(
                ImageRef(
                    alt_text=alt_text,
                    original_path=original_path,
                    source_path=str(source_path),
                    copied_path="",
                    markdown_path=original_path,
                )
            )
            return match.group(0)

        safe_name = f"chunk_{chunk_index}_{source_path.name}"
        copied_path = images_output_dir / safe_name

        shutil.copy2(source_path, copied_path)

        markdown_path = f"images/{safe_name}"

        images.append(
            ImageRef(
                alt_text=alt_text,
                original_path=original_path,
                source_path=str(source_path),
                copied_path=str(copied_path.resolve()),
                markdown_path=markdown_path,
            )
        )

        return f"![{alt_text}]({markdown_path})"

    rewritten_content = image_pattern.sub(replace_image, content)

    return rewritten_content, images


def split_by_exercise_pattern(
    markdown_text: str,
    chapter: Optional[str],
    md_file_path: Path,
    output_dir: Path
) -> list[Chunk]:

    section_pattern = re.compile(
        r'^#{1,3}\s*'
        r'(Exercise|EXERCISE|Example|EXAMPLE)'
        r'\s+'
        r'([\d]+(?:\.[\d]+)?)'
        r'[^\n]*$',
        re.MULTILINE
    )

    matches = list(section_pattern.finditer(markdown_text))

    if not matches:
        return []

    chunks: list[Chunk] = []

    for i, match in enumerate(matches, start=1):
        label_word = match.group(1).capitalize()
        num = match.group(2)

        label = f"{label_word} {num}"
        chunk_type = "exercise" if label_word.lower() == "exercise" else "example"

        start = match.start()
        end = matches[i].start() if i < len(matches) else len(markdown_text)

        raw_content = markdown_text[start:end].strip()

        content, images = copy_images_and_rewrite_content(
            raw_content,
            md_file_path,
            output_dir,
            i
        )

        chunks.append(
            Chunk(
                label=label,
                chunk_type=chunk_type,
                exercise_num=num,
                content=content,
                chapter=chapter,
                images=images,
            )
        )

    return chunks


def split_by_named_sections(
    markdown_text: str,
    chapter: Optional[str],
    md_file_path: Path,
    output_dir: Path
) -> list[Chunk]:

    named_section_pattern = re.compile(
        r'^(?:#{1,4}\s+(.+)|'
        r'\*{1,2}([A-Z][^*\n]{2,})\*{1,2})'
        r'\s*$',
        re.MULTILINE
    )

    matches = list(named_section_pattern.finditer(markdown_text))

    if not matches:
        return []

    chunks: list[Chunk] = []
    chunk_number = 1

    for i, match in enumerate(matches):
        label = (match.group(1) or match.group(2) or "").strip()

        if not label or len(label) < 3:
            continue

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)

        raw_content = markdown_text[start:end].strip()

        if len(raw_content) < 80:
            continue

        content, images = copy_images_and_rewrite_content(
            raw_content,
            md_file_path,
            output_dir,
            chunk_number
        )

        chunks.append(
            Chunk(
                label=label,
                chunk_type="section",
                exercise_num="",
                content=content,
                chapter=chapter,
                images=images,
            )
        )

        chunk_number += 1

    return chunks


def split_as_full_chapter(
    markdown_text: str,
    chapter: Optional[str],
    md_file_path: Path,
    output_dir: Path
) -> list[Chunk]:

    raw_content = markdown_text.strip()

    content, images = copy_images_and_rewrite_content(
        raw_content,
        md_file_path,
        output_dir,
        1
    )

    return [
        Chunk(
            label=chapter or "Full chapter",
            chunk_type="full_chapter",
            exercise_num="",
            content=content,
            chapter=chapter,
            images=images,
        )
    ]


def split_into_chunks(
    markdown_text: str,
    md_file_path: Path,
    output_dir: Path
) -> list[Chunk]:

    chapter = extract_chapter(markdown_text)

    chunks = split_by_exercise_pattern(
        markdown_text,
        chapter,
        md_file_path,
        output_dir
    )

    if chunks:
        print(f"Strategy used: Exercise/Example headers → {len(chunks)} chunks")
        return chunks

    chunks = split_by_named_sections(
        markdown_text,
        chapter,
        md_file_path,
        output_dir
    )

    if chunks:
        print(f"Strategy used: Named section headers → {len(chunks)} chunks")
        return chunks

    print("Strategy used: Full chapter fallback → 1 chunk")

    return split_as_full_chapter(
        markdown_text,
        chapter,
        md_file_path,
        output_dir
    )


def save_chunks(chunks: list[Chunk], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "chunks.json"
    md_path = output_dir / "chunks.md"

    chunks_data = [asdict(chunk) for chunk in chunks]

    json_path.write_text(
        json.dumps(chunks_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    md_lines: list[str] = [
        "# Extracted QA Chunks",
        "",
        f"Total chunks: {len(chunks)}",
        ""
    ]

    for index, chunk in enumerate(chunks, start=1):
        md_lines.append(f"## Chunk {index}: {chunk.label}")
        md_lines.append("")
        md_lines.append(f"- **Chunk Type:** {chunk.chunk_type}")
        md_lines.append(f"- **Exercise Number:** {chunk.exercise_num}")
        md_lines.append(f"- **Chapter:** {chunk.chapter}")
        md_lines.append(f"- **Images Found:** {len(chunk.images)}")
        md_lines.append("")

        if chunk.images:
            md_lines.append("### Images")
            md_lines.append("")
            for img_index, image in enumerate(chunk.images, start=1):
                md_lines.append(f"{img_index}. Alt text: `{image.alt_text}`")
                md_lines.append(f"   - Original Path: `{image.original_path}`")
                md_lines.append(f"   - Source Path: `{image.source_path}`")
                md_lines.append(f"   - Copied Path: `{image.copied_path}`")
                md_lines.append(f"   - Markdown Path: `{image.markdown_path}`")
            md_lines.append("")

        md_lines.append("### Content")
        md_lines.append("")
        md_lines.append(chunk.content)
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved chunks JSON    : {json_path}")
    print(f"Saved chunks Markdown: {md_path}")
    print(f"Saved images folder  : {output_dir / 'images'}")

    return json_path


# ─────────────────────────────────────────────
# Step 3: OpenAI question extraction
# ─────────────────────────────────────────────

MEDIA_TYPE_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def get_media_type(path_str: str) -> str:
    return MEDIA_TYPE_MAP.get(Path(path_str).suffix.lower(), "image/jpeg")


def encode_image(abs_path: str) -> Optional[str]:
    path = Path(abs_path)

    if not path.exists():
        print(f"    [WARN] Image not found: {abs_path}")
        return None

    return base64.b64encode(path.read_bytes()).decode("utf-8")


def get_image_path_from_ref(img: dict) -> str:
    return (
        img.get("copied_path")
        or img.get("absolute_path")
        or img.get("source_path")
        or img.get("path")
        or img.get("original_path")
        or ""
    )


def build_content_blocks(chunk: dict) -> list[dict]:
    content_blocks: list[dict] = []

    chunk_text = chunk.get("content", "")
    images = chunk.get("images", [])

    if chunk_text.strip():
        content_blocks.append({
            "type": "text",
            "text": f"""
Chunk Label: {chunk.get("label", "")}
Chunk Type: {chunk.get("chunk_type", "")}
Chapter: {chunk.get("chapter", "")}

Content:
{chunk_text}
"""
        })

    for img in images:
        image_path = get_image_path_from_ref(img)

        if not image_path:
            continue

        if image_path.startswith(("http://", "https://")):
            content_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": image_path
                }
            })
            continue

        b64 = encode_image(image_path)

        if not b64:
            continue

        media_type = get_media_type(image_path)

        content_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{b64}"
            }
        })

    return content_blocks


def safe_parse_json(text: str) -> dict:
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])

        raise


def call_openai(client: OpenAI, content_blocks: list[dict]) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": content_blocks
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    text = response.choices[0].message.content or ""

    return safe_parse_json(text)


def save_question_outputs(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    questions_json_path = output_dir / "questions.json"
    questions_flat_json_path = output_dir / "questions_flat.json"
    questions_md_path = output_dir / "questions.md"

    questions_json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    all_questions = []

    for result in results:
        for q in result.get("questions", []):
            all_questions.append({
                "chunk_label": result.get("chunk_label", ""),
                "chunk_type": result.get("chunk_type", ""),
                "chapter": result.get("chapter", ""),
                "image_count": result.get("image_count", 0),
                **q
            })

    questions_flat_json_path.write_text(
        json.dumps(all_questions, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    md_lines = [
        "# Extracted Questions",
        "",
        f"Total chunks processed: {len(results)}",
        f"Total questions extracted: {len(all_questions)}",
        "",
        "---",
        ""
    ]

    q_counter = 1

    for result in results:
        md_lines.append(f"## Chunk: {result.get('chunk_label', '')}")
        md_lines.append("")
        md_lines.append(f"- **Chapter:** {result.get('chapter', '')}")
        md_lines.append(f"- **Chunk Type:** {result.get('chunk_type', '')}")
        md_lines.append(f"- **Images:** {result.get('image_count', 0)}")
        md_lines.append("")

        if result.get("error"):
            md_lines.append(f"**Error:** {result['error']}")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
            continue

        questions = result.get("questions", [])

        if not questions:
            md_lines.append("_No questions found._")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
            continue

        for q in questions:
            md_lines.append(f"### Q{q_counter}. {q.get('question_text', '')}")
            md_lines.append("")
            md_lines.append(f"- **Original Number:** {q.get('question_number', '')}")
            md_lines.append(f"- **Question Type:** {q.get('question_type', '')}")
            md_lines.append(f"- **Has Image:** {q.get('has_image', False)}")

            sub_parts = q.get("sub_parts", [])

            if sub_parts:
                md_lines.append("")
                md_lines.append("**Sub-parts:**")
                for sp in sub_parts:
                    md_lines.append(f"- {sp}")

            md_lines.append("")
            q_counter += 1

        md_lines.append("---")
        md_lines.append("")

    questions_md_path.write_text(
        "\n".join(md_lines),
        encoding="utf-8"
    )

    print(f"Saved questions JSON     : {questions_json_path}")
    print(f"Saved questions flat JSON: {questions_flat_json_path}")
    print(f"Saved questions Markdown : {questions_md_path}")


def extract_questions_from_chunks(
    chunks_json_path: Path,
    questions_output_dir: Path
) -> None:

    load_dotenv()

    client = OpenAI()

    chunks = json.loads(chunks_json_path.read_text(encoding="utf-8"))

    print()
    print(f"Loaded chunks: {len(chunks)}")
    print(f"Input chunks file: {chunks_json_path}")
    print(f"Questions output folder: {questions_output_dir}")
    print()

    results = []

    for idx, chunk in enumerate(chunks, start=1):
        label = chunk.get("label", f"Chunk {idx}")
        chapter = chunk.get("chapter", "")
        chunk_type = chunk.get("chunk_type", "")
        image_count = len(chunk.get("images", []))

        print(f"[{idx}/{len(chunks)}] {label} | type={chunk_type} | images={image_count}")

        try:
            content_blocks = build_content_blocks(chunk)

            if not content_blocks:
                print("  → Skipped: empty chunk")
                continue

            raw_result = call_openai(client, content_blocks)
            questions = raw_result.get("questions", [])

            print(f"  → Questions extracted: {len(questions)}")

            results.append({
                "chunk_label": label,
                "chunk_type": chunk_type,
                "chapter": chapter,
                "image_count": image_count,
                "questions": questions,
            })

        except Exception as e:
            print(f"  [ERROR] {e}")

            results.append({
                "chunk_label": label,
                "chunk_type": chunk_type,
                "chapter": chapter,
                "image_count": image_count,
                "questions": [],
                "error": str(e),
            })

        if idx < len(chunks):
            time.sleep(DELAY_SECONDS)

    save_question_outputs(results, questions_output_dir)


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────

def main() -> None:
    print("Starting PDF → Markdown → Chunks → Questions pipeline")
    print(f"Input PDF: {INPUT_PDF}")
    print(f"Output folder: {BASE_OUTPUT_DIR}")
    print()

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_path = run_marker_if_needed(INPUT_PDF, MARKER_OUTPUT_DIR)

    markdown_text = md_path.read_text(encoding="utf-8")

    chunks = split_into_chunks(
        markdown_text=markdown_text,
        md_file_path=md_path,
        output_dir=CHUNKS_OUTPUT_DIR
    )

    print()
    print(f"Total chunks created: {len(chunks)}")
    print()

    for chunk in chunks:
        print(f"[{chunk.chunk_type.upper()}] {chunk.label}")
        print(f"Images found: {len(chunk.images)}")
        print(f"Preview: {chunk.content[:120].replace(chr(10), ' ')}...")
        print()

    chunks_json_path = save_chunks(chunks, CHUNKS_OUTPUT_DIR)

    extract_questions_from_chunks(
        chunks_json_path=chunks_json_path,
        questions_output_dir=QUESTIONS_OUTPUT_DIR
    )

    print()
    print("Pipeline complete.")
    print(f"Marker output    : {MARKER_OUTPUT_DIR}")
    print(f"Chunks output    : {CHUNKS_OUTPUT_DIR}")
    print(f"Questions output : {QUESTIONS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()