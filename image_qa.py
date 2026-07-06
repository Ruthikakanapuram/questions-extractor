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

# INPUT_PDF = Path("input/9th_hindi_c2.pdf")
# INPUT_PDF = Path("input/9th_science_c2.pdf")
INPUT_PDF = Path("input/9th_social_c2.pdf")
# INPUT_PDF = Path("input/9th_maths.pdf")

BASE_OUTPUT_DIR = Path("output")
PDF_OUTPUT_DIR = BASE_OUTPUT_DIR / INPUT_PDF.stem

MARKER_OUTPUT_DIR = PDF_OUTPUT_DIR / "marker_output"
CHUNKS_OUTPUT_DIR = PDF_OUTPUT_DIR / "chunks"
QUESTIONS_OUTPUT_DIR = PDF_OUTPUT_DIR / "questions"

MODEL = "gpt-4.1-mini"
DELAY_SECONDS = 1.0


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a precise question extraction tool for NCERT/CBSE textbook content across all subjects including Mathematics, Science, Social Studies, English, Biology, and Hindi.

Your ONLY job is to find student-facing questions in the provided textbook chunk and copy them EXACTLY as written. Never answer the questions.

## RULE 1 — VERBATIM EXTRACTION
- Copy every extracted question EXACTLY as it appears in the source.
- Do NOT rephrase, reword, summarize, correct grammar, fix spelling, or clean up text.
- Do NOT translate or transliterate any word.
- Preserve original symbols, blanks (___), punctuation, numbering, and formatting.

## RULE 2 — WHAT TO EXTRACT
Extract text that asks a student to DO or ANSWER something. This includes:
- Numbered problems: 1., 2., Q1., Q.1, Question 1, Ex. 1, etc.
- Image based questions: "Look at the picture", "Shown below", "Observe the figure", etc.
- Fill in the blank tasks
- Match the following / True / False / Observe and answer
- Activity prompts: "Try this", "Find out", "Discuss", "Do this", "Think and write"
- Puzzles or games with a clear task
- Sub-questions marked (a), (b), (c) or (i), (ii), (iii)
- Questions that depend on an image, figure, diagram, or picture

## RULE 3 — WHAT TO SKIP
Skip these:
- Definitions, explanations, body text
- Solved/worked examples
- "Note for Teachers", "For the Teacher", or teacher-facing content
- Rhetorical chapter-opening questions that are not subject-specific tasks
- Standalone table headers or labels with no task attached
- Figure captions that do not ask the student to do anything

## RULE 4 — GROUPING
- The procedure steps (numbered 1, 2, 3...) are NOT questions — do not extract them.
- Only extract the observation/conclusion/inference questions that follow the procedure.
Treat as ONE question when:
- A scenario/narrative is followed by blanks about the same context
- A question stem is followed by a table the student must fill
- A visual/image is followed by blanks or questions about that visual
- Sub-parts (a), (b), (i), (ii) follow a question stem

Do NOT split a question just because:
- It has multiple blanks / both text and image / both sentence and table

## RULE 5 — SHARED CONTEXT BLOCK (NEW)
Many exercises begin with a setup paragraph and/or figure that applies to ALL
questions that follow. This is called the "context block".

Identify a context block when:
- A paragraph appears BEFORE the numbered questions
- It sets up a scenario, coordinate system, map, table, or figure
- The numbered questions refer back to it ("Using Fig. 1.5", "From the above", etc.)

When a context block exists:
- Extract it verbatim and put it in "context" on EVERY question that depends on it.
- Set "has_image": true on every question if the context block contains a figure/image.
- Include the figure label (e.g., "Fig. 1.5") in the context text.
- Do NOT drop the context just because it appears before question numbering starts.

Example context block from a chunk:
  "On a graph sheet, mark the x-axis and y-axis... (Use the scale 1 cm = 1 unit.)
   Using Fig. 1.5, answer the given questions. [Fig. 1.5 image]"
→ This must appear in "context" for questions 1, 2, 3, and 4.

## RULE 6 — IMAGE HANDLING
- If a question or its context block contains or references a figure/image/diagram,
  set has_image: true.
- If the text says "Using Fig. X", "Observe the figure", "Shown below", or similar,
  set has_image: true even if you cannot see the image directly.
- Do not invent image details. Just flag the dependency.

## RULE 7 — question_number
- Use the number/label exactly as printed.
- If no number is printed, use "—".
- Do NOT create your own numbering.

## RULE 8 — sub_parts
- If a question has sub-parts, include them in sub_parts.
- Also keep the sub-parts inside question_text.
- If no sub-parts exist, return "sub_parts": [].

## RULE 9 — image_refs
When a question depends on an image, figure, picture, map, or diagram:
- Set has_image: true.
- Add the related image id in image_refs.
- Image ids will be provided as [IMAGE_REF: image_1], [IMAGE_REF: image_2], etc.
- If one image is shared by a context block for many questions, include the same image id in every dependent question.
- If no image is needed, return "image_refs": [].
## RULE 10 — MULTI-IMAGE ANALYSIS PASS (do this BEFORE extracting questions)

When a chunk contains more than one image, follow this exact reasoning sequence:

STEP 1 — INVENTORY ALL IMAGES
Before extracting any question, list every image in the chunk in reading order.
For each image note:
  - Its IMAGE_REF id (e.g., image_1, image_2)
  - Its position in the chunk (before/after which text)
  - What it visually contains: diagram, graph, map, figure, table, decorative art, icon, chapter illustration
  - Whether it has a label near it in the text (e.g., "Fig. 2.1", "Fig. 3", "Diagram A")

STEP 2 — CLASSIFY EACH IMAGE
For each image, decide:
  - QUESTION_IMAGE: directly required to answer one or more questions
      → the question cannot be understood or answered without this image
      → examples: a labelled diagram to identify parts, a graph to read values from,
        a coordinate figure to plot points on, a map to locate regions on
  - CONTEXT_IMAGE: part of a shared setup block that scopes multiple questions
      → example: Fig. 1.5 showing a floor plan used by questions 1–4
  - DECORATIVE: chapter artwork, section dividers, icons, portraits, unrelated illustrations
      → these must NEVER appear in image_refs

STEP 3 — BUILD AN IMAGE-TO-QUESTION MAP
For each QUESTION_IMAGE or CONTEXT_IMAGE:
  - Identify which question number(s) depend on it
  - Use these signals to decide dependency:
      a) The question text explicitly names the figure: "Using Fig. 2.1", "Refer to the diagram above"
      b) The question text uses proximity language: "shown below", "above figure", "observe the picture"
         → map to the nearest image in reading order (the image immediately before or after)
      c) The image appears inside a context block that scopes questions 1–N
         → map that image to ALL questions 1–N
      d) The question asks to label, draw, identify, or plot something
         and an image immediately precedes or follows it
         → map that image to that question
      e) No question refers to the image AND it is not part of a context block
         → classify as DECORATIVE, do not map

STEP 4 — VALIDATE BEFORE ASSIGNING
Before adding an image_ref to any question, confirm:
  ✓ The question cannot be fully understood without the image
  ✓ The image is the correct one (matches figure label or is nearest in reading order)
  ✓ The image is not decorative or illustrative-only
  ✓ You are not attaching an image just because it is nearby —
    proximity alone is not sufficient; the question must actually need it

Only after completing STEPS 1–4 should you begin extracting questions and populating image_refs.
## RULE 11 — IMAGE ASSIGNMENT SUMMARY
Apply the image-to-question map built in RULE 11 when populating image_refs.
- A question may have 0, 1, or multiple image_refs depending on the map.
- If the map shows an image is shared across a context block, assign it to every dependent question.
- Never assign an image classified as DECORATIVE.
- If after RULE 11 analysis you are still uncertain whether an image is needed, set image_refs: [] and has_image: false. Do not guess.
## IMPORTANT:
- Always verify that the image reference is correct and the image is actually needed for the question.
- If a question has a context block that contains an image, ensure that image is included in the question's image_refs.
- If Table id or table is mentioned in the question, include the corresponding table also.
## OUTPUT FORMAT
Return ONLY valid JSON. No markdown fences. No commentary.

{
  "questions": [
    {
      "question_number": "<number or label as printed>",
      "context": "<verbatim setup paragraph and/or figure label that scopes this question, or null if none>",
      "question_text": "<exact verbatim text of the question itself>",
      "sub_parts": ["(i) <exact text>", "(ii) <exact text>"],
      "has_image": false,
      "image_refs": ["image_1"],
      "question_type": "fill_in_blank | mcq | short_answer | activity | puzzle | arrange | true_false | match | observe_and_answer | other"
    }
  ]
}

If no valid questions are found, return:
{ "questions": [] }
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
    chunk_type: str
    exercise_num: str
    content: str
    chapter: Optional[str] = None
    images: list[ImageRef] = field(default_factory=list)


# ─────────────────────────────────────────────
# Step 1: Run Marker
# ─────────────────────────────────────────────

def run_marker_if_needed(input_pdf: Path, marker_output_dir: Path) -> Path:
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
# Step 2: Chunk extraction
# ─────────────────────────────────────────────
def build_chunk_image_refs(chunk: dict) -> list[dict]:
    """
    Create stable image ids for all images in a chunk.

    Example:
    image_1 -> images/chunk_10__page_6_Figure_10.jpeg
    """
    refs = []

    for index, img in enumerate(chunk.get("images", []), start=1):
        refs.append({
            "image_id": f"image_{index}",
            "alt_text": img.get("alt_text", ""),
            "original_path": img.get("original_path", ""),
            "source_path": img.get("source_path", ""),
            "copied_path": img.get("copied_path", ""),
            "markdown_path": img.get("markdown_path", ""),
        })

    return refs


def build_image_lookup_from_refs(image_refs: list[dict]) -> dict[str, dict]:
    """
    Build lookup so markdown image paths in chunk content can be matched
    with stable image refs.
    """
    lookup: dict[str, dict] = {}

    for ref in image_refs:
        for key in [
            "markdown_path",
            "original_path",
            "copied_path",
            "source_path",
        ]:
            value = ref.get(key)
            if value:
                lookup[value] = ref
                lookup[Path(value).name] = ref

    return lookup


# def resolve_question_image_refs(question: dict, chunk_image_refs: list[dict]) -> list[dict]:
#     """
#     Convert model-returned image_refs like ["image_1"] into full image metadata.

#     If has_image is true but model forgot image_refs, attach all chunk images.
#     This is useful when a shared figure applies to all questions in the chunk.
#     """
#     if not question.get("has_image"):
#         return []

#     refs_by_id = {
#         ref["image_id"]: ref
#         for ref in chunk_image_refs
#     }

#     raw_refs = question.get("image_refs", [])

#     resolved = []

#     if isinstance(raw_refs, list):
#         for item in raw_refs:
#             if isinstance(item, str):
#                 ref = refs_by_id.get(item)
#                 if ref:
#                     resolved.append(ref)

#             elif isinstance(item, dict):
#                 image_id = item.get("image_id")
#                 ref = refs_by_id.get(image_id)
#                 if ref:
#                     resolved.append(ref)

#     # Fallback: if question depends on image but model did not return image_refs,
#     # attach all images from that chunk.
#     if not resolved and chunk_image_refs:
#         resolved = chunk_image_refs

#     return resolved

def resolve_question_image_refs(question: dict, chunk_image_refs: list[dict]) -> list[dict]:
    """
    Convert model-returned image_refs like ["image_1"] into full image metadata.

    Safer behavior:
    - Only attach images explicitly returned by the model.
    - Do NOT attach all chunk images automatically.
    - If has_image is true but image_refs is empty, keep it empty and mark warning.
    """

    if not question.get("has_image"):
        return []

    refs_by_id = {
        ref["image_id"]: ref
        for ref in chunk_image_refs
    }

    raw_refs = question.get("image_refs", [])
    resolved = []

    if isinstance(raw_refs, list):
        for item in raw_refs:
            if isinstance(item, str):
                ref = refs_by_id.get(item)
                if ref:
                    resolved.append(ref)

            elif isinstance(item, dict):
                image_id = item.get("image_id")
                ref = refs_by_id.get(image_id)
                if ref:
                    resolved.append(ref)

    # Do NOT blindly attach all images.
    # This avoids wrong mapping when a chunk has multiple images.
    if question.get("has_image") and not resolved:
        question["image_mapping_warning"] = (
            "Model marked this question as image-based but did not provide a valid image_ref."
        )

    return resolved
def extract_chapter(text: str) -> Optional[str]:
    match = re.search(
        r'^#{1,3}\s*(Chapter\s+\d+[^\n]*|CHAPTER\s+\d+[^\n]*)',
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

# ─────────────────────────────────────────────
# Better chunk extraction strategy
# Unified splitter for:
# - Exercise / Exercise Set
# - Example
# - Let Us Think / Let Us Do / Try This / Find Out
# - Numbered concept headings like 1.4 Distance Between Two Points
# - Normal markdown headings
#
# It ignores:
# - Fig. 1.5
# - Figure 1.5
# - Table 1.2
# - image captions
# ─────────────────────────────────────────────

def clean_heading_label(label: str) -> str:
    """
    Clean markdown heading label.
    Example:
    **Exercise Set 1.2** -> Exercise Set 1.2
    *Fig. 1.5* -> Fig. 1.5
    """
    label = label.strip()
    label = label.strip("#").strip()
    label = label.strip("*").strip()
    label = re.sub(r"<[^>]+>", "", label)
    label = re.sub(r"\s+", " ", label)
    return label.strip()


def is_caption_or_noise_heading(label: str) -> bool:
    """
    These should NOT become chunk headings.
    They should stay inside the current chunk.
    """
    clean = clean_heading_label(label)

    noise_patterns = [
        r"^(fig\.?|figure)\s*\d+(\.\d+)*$",
        r"^(table)\s*\d+(\.\d+)*$",
        r"^source\s*:?",
        r"^image\s*$",
        r"^picture\s*$",
        r"^diagram\s*$",
        r"^caption\s*:?",
    ]

    for pattern in noise_patterns:
        if re.match(pattern, clean, re.IGNORECASE):
            return True

    return False


def classify_heading(label: str) -> str:
    """
    Decide what type of chunk heading it is.
    """
    clean = clean_heading_label(label)

    if re.match(r"^exercise(\s+set)?\s+\d+(\.\d+)*", clean, re.IGNORECASE):
        return "exercise"

    if re.match(r"^example\s+\d+(\.\d+)*", clean, re.IGNORECASE):
        return "example"

    activity_patterns = [
        r"^let'?s\s+",
        r"^let\s+us\s+",
        r"^try\s+this",
        r"^do\s+this",
        r"^find\s+out",
        r"^think\s+and\s+write",
        r"^discuss",
        r"^activity",
        r"^practice",
        r"^worksheet",
        r"^puzzle",
        r"^game",
        r"^maths\s+lab",
        r"^explore",
        r"^observe",
    ]

    for pattern in activity_patterns:
        if re.match(pattern, clean, re.IGNORECASE):
            return "activity"

    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", clean):
        return "concept"

    return "section"


def is_valid_chunk_heading(label: str) -> bool:
    """
    Decide whether a heading should create a new chunk.
    """
    clean = clean_heading_label(label)

    if not clean or len(clean) < 3:
        return False

    if is_caption_or_noise_heading(clean):
        return False

    # Avoid headings that are only punctuation/numbers
    if re.match(r"^[\W\d_]+$", clean):
        return False

    return True


def find_all_chunk_headings(markdown_text: str) -> list[tuple[re.Match, str, str]]:

    heading_pattern = re.compile(
        r"^("
        r"#{1,6}\s+(.+?)"                    # markdown heading
        r"|"
        r"\*\*([A-Za-z0-9][^*\n]{2,})\*\*"   # bold heading only
        r")\s*$",
        re.MULTILINE
    )

    headings = []

    for match in heading_pattern.finditer(markdown_text):
        raw_label = match.group(2) or match.group(3) or ""
        label = clean_heading_label(raw_label)

        if not is_valid_chunk_heading(label):
            continue

        chunk_type = classify_heading(label)

        headings.append((match, label, chunk_type))

    return headings


def split_by_best_heading_strategy(
    markdown_text: str,
    chapter: Optional[str],
    md_file_path: Path,
    output_dir: Path
) -> list[Chunk]:
    """
    Best strategy:
    Create chunks from all meaningful textbook headings,
    not only exercises/examples.
    """
    headings = find_all_chunk_headings(markdown_text)

    if not headings:
        return []

    chunks: list[Chunk] = []
    chunk_number = 1

    for i, (match, label, chunk_type) in enumerate(headings):
        start = match.start()
        end = headings[i + 1][0].start() if i + 1 < len(headings) else len(markdown_text)

        raw_content = markdown_text[start:end].strip()

        if len(raw_content) < 80:
            continue

        content, images = copy_images_and_rewrite_content(
            raw_content,
            md_file_path,
            output_dir,
            chunk_number
        )

        exercise_num = ""

        exercise_match = re.search(
            r"(?:Exercise(?:\s+Set)?|Example)\s+(\d+(?:\.\d+)*)",
            label,
            re.IGNORECASE
        )

        if exercise_match:
            exercise_num = exercise_match.group(1)

        chunks.append(
            Chunk(
                label=label,
                chunk_type=chunk_type,
                exercise_num=exercise_num,
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

    chunks = split_by_best_heading_strategy(
        markdown_text=markdown_text,
        chapter=chapter,
        md_file_path=md_file_path,
        output_dir=output_dir
    )

    if chunks:
        print(f"Strategy used: Unified textbook heading strategy → {len(chunks)} chunks")
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


def build_image_lookup(images: list[dict]) -> dict[str, dict]:
    """
    Build lookup so markdown image paths in chunk content can be matched
    with the corresponding copied image file.

    Example content image:
      ![](images/chunk_1__page_0_Picture_7.jpeg)

    Matching image field:
      markdown_path = images/chunk_1__page_0_Picture_7.jpeg
    """
    lookup: dict[str, dict] = {}

    for img in images:
        for key in [
            "markdown_path",
            "original_path",
            "copied_path",
            "source_path",
            "absolute_path",
            "path",
        ]:
            value = img.get(key)
            if value:
                lookup[value] = img
                lookup[Path(value).name] = img

    return lookup


def make_image_block(image_path: str) -> Optional[dict]:
    if not image_path:
        return None

    if image_path.startswith(("http://", "https://")):
        return {
            "type": "image_url",
            "image_url": {
                "url": image_path
            }
        }

    b64 = encode_image(image_path)

    if not b64:
        return None

    media_type = get_media_type(image_path)

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};base64,{b64}"
        }
    }

def build_interleaved_content_blocks(chunk: dict) -> list[dict]:
    """
    Build OpenAI multimodal content in the SAME ORDER as chunk content.

    It also gives each image a stable id:
      image_1, image_2, image_3...

    The model can return these ids in question.image_refs.
    """
    content_blocks: list[dict] = []

    chunk_text = chunk.get("content", "")
    image_refs = build_chunk_image_refs(chunk)

    if not chunk_text.strip():
        return []

    header_text = f"""
Board: NCERT
Class: 9
Subject: Social

Chunk Label: {chunk.get("label", "")}
Chunk Type: {chunk.get("chunk_type", "")}
Chapter: {chunk.get("chapter", "")}

The following content blocks are in the same order as the textbook chunk.
Images are inserted exactly where their markdown image tag appeared.

IMPORTANT:
- Each image has an image id.
- When a question depends on an image, return that image id in image_refs.
- Example: "image_refs": ["image_1"]
"""
    content_blocks.append({
        "type": "text",
        "text": header_text.strip()
    })

    image_lookup = build_image_lookup_from_refs(image_refs)
    image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    last_end = 0

    for match in image_pattern.finditer(chunk_text):
        before_text = chunk_text[last_end:match.start()].strip()

        if before_text:
            content_blocks.append({
                "type": "text",
                "text": before_text
            })

        alt_text = match.group(1).strip()
        markdown_image_path = match.group(2).strip().strip('"').strip("'")

        img_ref = (
            image_lookup.get(markdown_image_path)
            or image_lookup.get(Path(markdown_image_path).name)
        )

        if img_ref:
            image_id = img_ref["image_id"]
            image_path = img_ref.get("copied_path") or img_ref.get("source_path")
            image_block = make_image_block(image_path)

            # content_blocks.append({
            #     "type": "text",
            #     "text": (
            #         f"[IMAGE_REF: {image_id}]\n"
            #         f"Markdown path: {img_ref.get('markdown_path', '')}\n"
            #         f"Original path: {img_ref.get('original_path', '')}\n"
            #         f"This image appears here in the textbook."
            #     )
            # })
            content_blocks.append({
                "type": "text",
                "text": (
                    f"\n[IMAGE_REF: {image_id}]\n"
                    f"This image appears at this exact position in the textbook chunk.\n"
                    f"Use this image id only for questions that refer to this nearby image, figure, diagram, map, table, or picture.\n"
                    f"Markdown path: {img_ref.get('markdown_path', '')}\n"
                    f"Original path: {img_ref.get('original_path', '')}\n"
                )
            })

            if image_block:
                content_blocks.append(image_block)
            else:
                content_blocks.append({
                    "type": "text",
                    "text": f"[Image {image_id} could not be loaded from disk.]"
                })

        else:
            content_blocks.append({
                "type": "text",
                "text": f"[Image appears here but metadata was not found: {markdown_image_path}]"
            })

        last_end = match.end()

    remaining_text = chunk_text[last_end:].strip()

    if remaining_text:
        content_blocks.append({
            "type": "text",
            "text": remaining_text
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
    """
    Important:
    The user message content must be content_blocks directly.
    Do NOT convert content_blocks to a string.
    Otherwise images will not be sent to the model.
    """
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

            image_refs = q.get("image_refs", [])

            if image_refs:
                md_lines.append("")
                md_lines.append("**Related Image(s):**")
                md_lines.append("")

                for img_ref in image_refs:
                    image_id = img_ref.get("image_id", "")
                    markdown_path = img_ref.get("markdown_path", "")
                    copied_path = img_ref.get("copied_path", "")

                    # questions.md is inside output/<pdf>/questions/
                    # chunk images are inside output/<pdf>/chunks/images/
                    if markdown_path:
                        relative_question_md_path = f"../chunks/{markdown_path}"
                        md_lines.append(f"- `{image_id}`: `{relative_question_md_path}`")
                        md_lines.append(f"")
                        md_lines.append(f"![{image_id}]({relative_question_md_path})")
                        md_lines.append("")
                    else:
                        md_lines.append(f"- `{image_id}`: `{copied_path}`")

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
            content_blocks = build_interleaved_content_blocks(chunk)

            if not content_blocks:
                print("  → Skipped: empty chunk")
                continue

            print(f"  → Sending {len(content_blocks)} content blocks to OpenAI")

            raw_result = call_openai(client, content_blocks)
            questions = raw_result.get("questions", [])
            chunk_image_refs = build_chunk_image_refs(chunk)

            for q in questions:
                q["image_refs"] = resolve_question_image_refs(q, chunk_image_refs)

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
    print(f"PDF output folder: {PDF_OUTPUT_DIR}")
    print()

    PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_path = run_marker_if_needed(INPUT_PDF, MARKER_OUTPUT_DIR)

    markdown_text = md_path.read_text(encoding="utf-8")

    chunks = split_into_chunks(
        markdown_text=markdown_text,
        md_file_path=md_path,
        output_dir=CHUNKS_OUTPUT_DIR
    )

    chunks_json_path = save_chunks(chunks, CHUNKS_OUTPUT_DIR)

    extract_questions_from_chunks(
        chunks_json_path=chunks_json_path,
        questions_output_dir=QUESTIONS_OUTPUT_DIR
    )

    print()
    print("Pipeline complete.")
    print(f"PDF output      : {PDF_OUTPUT_DIR}")
    print(f"Marker output   : {MARKER_OUTPUT_DIR}")
    print(f"Chunks output   : {CHUNKS_OUTPUT_DIR}")
    print(f"Questions output: {QUESTIONS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()