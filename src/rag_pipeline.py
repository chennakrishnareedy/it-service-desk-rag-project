"""
Core RAG pipeline for the IT Service Desk Assistant.

Components:
    - DocumentLoader : extract text from PDF / DOCX / HTML / TXT
    - chunk_text     : split documents into overlapping chunks
    - RagIndex       : embed chunks + FAISS vector store + retrieval
    - Guardrails     : prompt-injection detection + grounding/hallucination check
    - Answerer       : local LLM answer generation with citations + escalation
    - RagAssistant   : end-to-end orchestration

Everything runs locally and free (no API keys required).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL", "google/flan-t5-base")
TOP_K = int(os.environ.get("TOP_K", "4"))
# Minimum retrieval similarity (cosine) for an answer to be considered grounded.
MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0.25"))


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A retrievable chunk of a source document."""

    text: str
    source: str
    page: int
    chunk_id: int


class DocumentLoader:
    """Extract text from PDF, DOCX, HTML and TXT files."""

    @staticmethod
    def load_pdf(path: Path) -> List[tuple[int, str]]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            pages.append((i, page.extract_text() or ""))
        return pages

    @staticmethod
    def load_docx(path: Path) -> List[tuple[int, str]]:
        from docx import Document

        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        return [(1, text)]

    @staticmethod
    def load_html(path: Path) -> List[tuple[int, str]]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return [(1, soup.get_text(separator="\n"))]

    @staticmethod
    def load_txt(path: Path) -> List[tuple[int, str]]:
        return [(1, path.read_text(encoding="utf-8"))]

    @classmethod
    def load(cls, path: Path) -> List[tuple[int, str]]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return cls.load_pdf(path)
        if suffix == ".docx":
            return cls.load_docx(path)
        if suffix in (".html", ".htm"):
            return cls.load_html(path)
        if suffix == ".txt":
            return cls.load_txt(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")


def chunk_text(
    text: str, source: str, page: int, start_id: int, size: int = 120, overlap: int = 30
) -> List[Chunk]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks: List[Chunk] = []
    if not words:
        return chunks
    step = max(1, size - overlap)
    cid = start_id
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        chunk_str = " ".join(window).strip()
        if chunk_str:
            chunks.append(Chunk(text=chunk_str, source=source, page=page, chunk_id=cid))
            cid += 1
        if start + size >= len(words):
            break
    return chunks


def build_chunks(docs_dir: Path) -> List[Chunk]:
    """Load every supported document in a directory and chunk it."""
    chunks: List[Chunk] = []
    cid = 0
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in (".pdf", ".docx", ".html", ".htm", ".txt"):
            continue
        for page, text in DocumentLoader.load(path):
            new = chunk_text(text, source=path.name, page=page, start_id=cid)
            chunks.extend(new)
            cid += len(new)
    return chunks


# ---------------------------------------------------------------------------
# Vector index (FAISS)
# ---------------------------------------------------------------------------
@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class RagIndex:
    """Embed chunks with sentence-transformers and search with FAISS."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.index = None

    def build(self, chunks: List[Chunk]) -> None:
        import faiss

        self.chunks = chunks
        embeddings = self.model.encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        dim = embeddings.shape[1]
        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = TOP_K) -> List[Retrieved]:
        q = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        scores, idx = self.index.search(q, top_k)
        results: List[Retrieved] = []
        for score, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            results.append(Retrieved(chunk=self.chunks[i], score=float(score)))
        return results


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore (all |previous |the )?(instructions|rules)",
    r"disregard (the |all |previous )?(instructions|rules|context)",
    r"forget (your |all |previous )?(instructions|rules)",
    r"reveal (your |the )?(system )?prompt",
    r"you are now",
    r"act as (an?|if)",
    r"developer mode",
    r"jailbreak",
    r"bypass (the |your )?(guardrails|safety|filters)",
]


@dataclass
class GuardResult:
    blocked: bool
    reason: str = ""


class Guardrails:
    """Prompt-injection detection and grounding (anti-hallucination) checks."""

    @staticmethod
    def check_injection(query: str) -> GuardResult:
        low = query.lower()
        for pat in INJECTION_PATTERNS:
            if re.search(pat, low):
                return GuardResult(
                    blocked=True,
                    reason="Request looks like a prompt-injection attempt and was refused.",
                )
        return GuardResult(blocked=False)

    @staticmethod
    def is_grounded(retrieved: List[Retrieved], min_similarity: float = MIN_SIMILARITY) -> bool:
        """Answer is grounded only if at least one chunk is sufficiently similar."""
        if not retrieved:
            return False
        return retrieved[0].score >= min_similarity


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------
# Escalation is driven by the user's intent/situation, not by incidental
# mentions of "Level 2" in the retrieved documentation.
ESCALATION_TRIGGERS = [
    "escalate",
    "level 2",
    "level two",
    "manual unlock",
    "account locked",
    "account is locked",
    "security exception",
    "vpn-500",
    "still not working",
    "still fails",
    "not resolved",
    "tried everything",
]


def should_escalate(query: str, retrieved: List[Retrieved], grounded: bool) -> Optional[str]:
    """Decide whether to route the issue to the support team.

    Escalation fires when:
      - the question cannot be answered from approved docs (route to Level 1), or
      - the user's own wording matches a known escalation situation.
    """
    if not grounded:
        return (
            "No approved documentation covers this question. "
            "Routing to the IT support team (Level 1)."
        )
    q = query.lower()
    for trigger in ESCALATION_TRIGGERS:
        if trigger in q:
            return (
                "This scenario matches a Level 2 escalation condition in the approved docs. "
                "Please raise a Level 2 support ticket."
            )
    return None


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are an IT Service Desk assistant. Answer the user's question using ONLY the context below.
If the context does not contain the answer, say you don't know.
Give clear step-by-step instructions when relevant.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class Answer:
    text: str
    citations: List[str] = field(default_factory=list)
    escalated: bool = False
    blocked: bool = False
    grounded: bool = True
    retrieved: List[Retrieved] = field(default_factory=list)


class Answerer:
    """Local LLM (flan-t5) answer generation."""

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question: str, retrieved: List[Retrieved]) -> str:
        context = "\n\n".join(
            f"[{r.chunk.source} p.{r.chunk.page}] {r.chunk.text}" for r in retrieved
        )
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        output = self.model.generate(
            **inputs, max_new_tokens=256, num_beams=4, no_repeat_ngram_size=3
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
class RagAssistant:
    """End-to-end IT Service Desk RAG assistant."""

    def __init__(self, docs_dir: Path, load_llm: bool = True):
        self.docs_dir = Path(docs_dir)
        self.index = RagIndex()
        self.index.build(build_chunks(self.docs_dir))
        self.answerer = Answerer() if load_llm else None

    def ask(self, question: str, top_k: int = TOP_K) -> Answer:
        # 1) Prompt-injection guardrail
        guard = Guardrails.check_injection(question)
        if guard.blocked:
            return Answer(text=guard.reason, blocked=True, grounded=False)

        # 2) Retrieve
        retrieved = self.index.search(question, top_k=top_k)

        # 3) Grounding / hallucination guardrail
        grounded = Guardrails.is_grounded(retrieved)

        # 4) Escalation workflow
        escalation = should_escalate(question, retrieved, grounded)

        if not grounded:
            return Answer(
                text=(
                    "I could not find this in the approved IT documentation. "
                    + (escalation or "")
                ),
                escalated=True,
                grounded=False,
                retrieved=retrieved,
            )

        # 5) Generate grounded answer
        if self.answerer is None:
            body = "\n\n".join(r.chunk.text for r in retrieved[:2])
        else:
            body = self.answerer.generate(question, retrieved)

        citations = []
        seen = set()
        for r in retrieved:
            tag = f"{r.chunk.source} (p.{r.chunk.page})"
            if tag not in seen:
                citations.append(tag)
                seen.add(tag)

        text = body
        escalated = False
        if escalation:
            text += f"\n\n⚠️ Escalation: {escalation}"
            escalated = True

        return Answer(
            text=text,
            citations=citations,
            escalated=escalated,
            grounded=True,
            retrieved=retrieved,
        )
