"""
SQLite database operations for the Researcher module.

Provides storage and querying for sources, findings, concepts, and numbers.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from .models import (
    Source, Finding, Concept, NumberMention, CrossReference,
    FindingType, MentionType, SourceTier
)


class ResearcherDatabase:
    """SQLite database for researcher findings and sources."""

    def __init__(self, db_path: Path):
        """Initialize database connection."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript("""
                -- Sources table
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    domain TEXT NOT NULL,
                    title TEXT,
                    trust_score INTEGER DEFAULT 0,
                    trust_tier INTEGER DEFAULT 0,
                    evaluation_reasoning TEXT,
                    evaluated_at TEXT,
                    content_hash TEXT,
                    content_type TEXT DEFAULT 'html',
                    last_fetched TEXT,
                    fetch_count INTEGER DEFAULT 1,
                    has_sanskrit_citations INTEGER DEFAULT 0,
                    has_verse_references INTEGER DEFAULT 0,
                    has_bibliography INTEGER DEFAULT 0,
                    is_academic_domain INTEGER DEFAULT 0,
                    topic_trust TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Findings table
                CREATE TABLE IF NOT EXISTS findings (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    finding_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    original_quote TEXT,
                    source_location TEXT,
                    relevance_score REAL DEFAULT 0,
                    relevance_reasoning TEXT,
                    confidence REAL DEFAULT 0.5,
                    extracted_at TEXT,
                    extraction_model TEXT,
                    domain TEXT,
                    FOREIGN KEY (source_id) REFERENCES sources(id)
                );

                -- Concepts table
                CREATE TABLE IF NOT EXISTS concepts (
                    id TEXT PRIMARY KEY,
                    canonical_name TEXT UNIQUE NOT NULL,
                    display_name TEXT,
                    aliases TEXT DEFAULT '[]',
                    domain TEXT,
                    sanskrit_forms TEXT DEFAULT '[]',
                    description TEXT,
                    related_concepts TEXT DEFAULT '[]',
                    occurrence_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Number mentions table
                CREATE TABLE IF NOT EXISTS number_mentions (
                    id TEXT PRIMARY KEY,
                    value INTEGER NOT NULL,
                    context TEXT,
                    mention_type TEXT,
                    significance TEXT,
                    finding_id TEXT,
                    source_id TEXT,
                    domain TEXT,
                    domains_found_in TEXT DEFAULT '[]',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (finding_id) REFERENCES findings(id),
                    FOREIGN KEY (source_id) REFERENCES sources(id)
                );

                -- Finding-concept junction table
                CREATE TABLE IF NOT EXISTS finding_concepts (
                    finding_id TEXT NOT NULL,
                    concept_id TEXT NOT NULL,
                    PRIMARY KEY (finding_id, concept_id),
                    FOREIGN KEY (finding_id) REFERENCES findings(id),
                    FOREIGN KEY (concept_id) REFERENCES concepts(id)
                );

                -- Cross-references table
                CREATE TABLE IF NOT EXISTS cross_references (
                    id TEXT PRIMARY KEY,
                    finding_id_1 TEXT NOT NULL,
                    finding_id_2 TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    relationship_value TEXT,
                    strength REAL DEFAULT 0.5,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (finding_id_1) REFERENCES findings(id),
                    FOREIGN KEY (finding_id_2) REFERENCES findings(id)
                );

                -- Indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_sources_domain ON sources(domain);
                CREATE INDEX IF NOT EXISTS idx_sources_trust ON sources(trust_score);
                CREATE INDEX IF NOT EXISTS idx_findings_source ON findings(source_id);
                CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(finding_type);
                CREATE INDEX IF NOT EXISTS idx_findings_domain ON findings(domain);
                CREATE INDEX IF NOT EXISTS idx_number_mentions_value ON number_mentions(value);
                CREATE INDEX IF NOT EXISTS idx_number_mentions_domain ON number_mentions(domain);
                CREATE INDEX IF NOT EXISTS idx_concepts_canonical ON concepts(canonical_name);
            """)

    # --- Source Operations ---

    def save_source(self, source: Source) -> None:
        """Save or update a source."""
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sources
                (id, url, domain, title, trust_score, trust_tier, evaluation_reasoning,
                 evaluated_at, content_hash, content_type, last_fetched, fetch_count,
                 has_sanskrit_citations, has_verse_references, has_bibliography,
                 is_academic_domain, topic_trust, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                source.id, source.url, source.domain, source.title,
                source.trust_score, source.trust_tier.value, source.evaluation_reasoning,
                source.evaluated_at.isoformat() if source.evaluated_at else None,
                source.content_hash, source.content_type,
                source.last_fetched.isoformat(), source.fetch_count,
                int(source.has_sanskrit_citations), int(source.has_verse_references),
                int(source.has_bibliography), int(source.is_academic_domain),
                json.dumps(source.topic_trust), source.created_at.isoformat()
            ))

    def get_source_by_url(self, url: str) -> Optional[Source]:
        """Get a source by URL."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE url = ?", (url,)
            ).fetchone()
            if row:
                return self._row_to_source(row)
            return None

    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE id = ?", (source_id,)
            ).fetchone()
            if row:
                return self._row_to_source(row)
            return None

    def get_trusted_sources(self, min_score: int = 70) -> list[Source]:
        """Get sources above a trust threshold."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sources WHERE trust_score >= ? ORDER BY trust_score DESC",
                (min_score,)
            ).fetchall()
            return [self._row_to_source(row) for row in rows]

    def _row_to_source(self, row: sqlite3.Row) -> Source:
        """Convert a database row to a Source object."""
        return Source(
            id=row["id"],
            url=row["url"],
            domain=row["domain"],
            title=row["title"],
            trust_score=row["trust_score"],
            trust_tier=SourceTier(row["trust_tier"]),
            evaluation_reasoning=row["evaluation_reasoning"] or "",
            evaluated_at=datetime.fromisoformat(row["evaluated_at"]) if row["evaluated_at"] else None,
            content_hash=row["content_hash"],
            content_type=row["content_type"],
            last_fetched=datetime.fromisoformat(row["last_fetched"]),
            fetch_count=row["fetch_count"],
            has_sanskrit_citations=bool(row["has_sanskrit_citations"]),
            has_verse_references=bool(row["has_verse_references"]),
            has_bibliography=bool(row["has_bibliography"]),
            is_academic_domain=bool(row["is_academic_domain"]),
            topic_trust=json.loads(row["topic_trust"]) if row["topic_trust"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # --- Finding Operations ---

    def save_finding(self, finding: Finding) -> None:
        """Save a finding."""
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO findings
                (id, source_id, finding_type, summary, original_quote, source_location,
                 relevance_score, relevance_reasoning, confidence, extracted_at,
                 extraction_model, domain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                finding.id, finding.source_id, finding.finding_type.value,
                finding.summary, finding.original_quote, finding.source_location,
                finding.relevance_score, finding.relevance_reasoning, finding.confidence,
                finding.extracted_at.isoformat(), finding.extraction_model, finding.domain
            ))

            # Update concept links
            for concept_id in finding.concepts:
                conn.execute("""
                    INSERT OR IGNORE INTO finding_concepts (finding_id, concept_id)
                    VALUES (?, ?)
                """, (finding.id, concept_id))

    def get_finding(self, finding_id: str) -> Optional[Finding]:
        """Get a finding by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM findings WHERE id = ?", (finding_id,)
            ).fetchone()
            if row:
                return self._row_to_finding(conn, row)
            return None

    def get_findings_by_domain(self, domain: str, limit: int = 100) -> list[Finding]:
        """Get findings for a domain."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM findings WHERE domain = ? ORDER BY extracted_at DESC LIMIT ?",
                (domain, limit)
            ).fetchall()
            return [self._row_to_finding(conn, row) for row in rows]

    def search_findings(
        self,
        concepts: list[str] = None,
        numbers: list[int] = None,
        domains: list[str] = None,
        min_trust: int = 50,
        min_confidence: float = 0.5,
        finding_types: list[str] = None,
        limit: int = 20
    ) -> list[Finding]:
        """Search findings with filters."""
        with self._connection() as conn:
            query = """
                SELECT DISTINCT f.* FROM findings f
                JOIN sources s ON f.source_id = s.id
                WHERE s.trust_score >= ? AND f.confidence >= ?
            """
            params = [min_trust, min_confidence]

            if domains:
                placeholders = ",".join("?" * len(domains))
                query += f" AND f.domain IN ({placeholders})"
                params.extend(domains)

            if finding_types:
                placeholders = ",".join("?" * len(finding_types))
                query += f" AND f.finding_type IN ({placeholders})"
                params.extend(finding_types)

            query += " ORDER BY f.relevance_score DESC, f.confidence DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            findings = [self._row_to_finding(conn, row) for row in rows]

            # Filter by concepts if specified
            if concepts:
                findings = [f for f in findings if any(c in f.concepts for c in concepts)]

            # Filter by numbers if specified
            if numbers:
                findings = [f for f in findings if any(n in f.numbers for n in numbers)]

            return findings[:limit]

    def _row_to_finding(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Finding:
        """Convert a database row to a Finding object."""
        # Get linked concepts
        concept_rows = conn.execute(
            "SELECT concept_id FROM finding_concepts WHERE finding_id = ?",
            (row["id"],)
        ).fetchall()
        concepts = [r["concept_id"] for r in concept_rows]

        # Get linked numbers
        number_rows = conn.execute(
            "SELECT id FROM number_mentions WHERE finding_id = ?",
            (row["id"],)
        ).fetchall()
        numbers = [r["id"] for r in number_rows]

        # Get related findings
        related_rows = conn.execute("""
            SELECT finding_id_2 FROM cross_references WHERE finding_id_1 = ?
            UNION
            SELECT finding_id_1 FROM cross_references WHERE finding_id_2 = ?
        """, (row["id"], row["id"])).fetchall()
        related = [r[0] for r in related_rows]

        return Finding(
            id=row["id"],
            source_id=row["source_id"],
            finding_type=FindingType(row["finding_type"]),
            summary=row["summary"],
            original_quote=row["original_quote"] or "",
            source_location=row["source_location"] or "",
            concepts=concepts,
            numbers=numbers,
            relevance_score=row["relevance_score"],
            relevance_reasoning=row["relevance_reasoning"] or "",
            confidence=row["confidence"],
            related_findings=related,
            extracted_at=datetime.fromisoformat(row["extracted_at"]),
            extraction_model=row["extraction_model"],
            domain=row["domain"],
        )

    # --- Concept Operations ---

    def save_concept(self, concept: Concept) -> None:
        """Save or update a concept."""
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO concepts
                (id, canonical_name, display_name, aliases, domain, sanskrit_forms,
                 description, related_concepts, occurrence_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                concept.id, concept.canonical_name, concept.display_name,
                json.dumps(concept.aliases), concept.domain,
                json.dumps(concept.sanskrit_forms), concept.description,
                json.dumps(concept.related_concepts), concept.occurrence_count,
                concept.created_at.isoformat()
            ))

    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Get a concept by canonical name or alias."""
        with self._connection() as conn:
            # Try canonical name first
            row = conn.execute(
                "SELECT * FROM concepts WHERE canonical_name = ?", (name.lower(),)
            ).fetchone()
            if row:
                return self._row_to_concept(row)

            # Search aliases
            rows = conn.execute("SELECT * FROM concepts").fetchall()
            for row in rows:
                aliases = json.loads(row["aliases"]) if row["aliases"] else []
                if name.lower() in [a.lower() for a in aliases]:
                    return self._row_to_concept(row)

            return None

    def get_or_create_concept(self, name: str, domain: str) -> Concept:
        """Get existing concept or create new one."""
        existing = self.get_concept_by_name(name)
        if existing:
            return existing

        concept = Concept.create(
            canonical_name=name.lower(),
            display_name=name,
            domain=domain
        )
        self.save_concept(concept)
        return concept

    def _row_to_concept(self, row: sqlite3.Row) -> Concept:
        """Convert a database row to a Concept object."""
        # Get finding IDs
        return Concept(
            id=row["id"],
            canonical_name=row["canonical_name"],
            display_name=row["display_name"],
            aliases=json.loads(row["aliases"]) if row["aliases"] else [],
            domain=row["domain"],
            sanskrit_forms=json.loads(row["sanskrit_forms"]) if row["sanskrit_forms"] else [],
            description=row["description"] or "",
            related_concepts=json.loads(row["related_concepts"]) if row["related_concepts"] else [],
            occurrence_count=row["occurrence_count"],
            finding_ids=[],  # Loaded separately if needed
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # --- Number Operations ---

    def save_number_mention(self, mention: NumberMention) -> None:
        """Save a number mention."""
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO number_mentions
                (id, value, context, mention_type, significance, finding_id,
                 source_id, domain, domains_found_in, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mention.id, mention.value, mention.context,
                mention.mention_type.value, mention.significance,
                mention.finding_id, mention.source_id, mention.domain,
                json.dumps(mention.domains_found_in), mention.created_at.isoformat()
            ))

    def get_number_occurrences(self, value: int) -> list[NumberMention]:
        """Get all occurrences of a number across domains."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM number_mentions WHERE value = ? ORDER BY created_at DESC",
                (value,)
            ).fetchall()
            return [self._row_to_number_mention(row) for row in rows]

    def get_number_summary(self, value: int) -> dict:
        """Get a summary of a number's appearances across domains."""
        with self._connection() as conn:
            # Get domains and counts
            rows = conn.execute("""
                SELECT domain, COUNT(*) as count,
                       GROUP_CONCAT(DISTINCT significance) as significances
                FROM number_mentions
                WHERE value = ?
                GROUP BY domain
            """, (value,)).fetchall()

            domains = {}
            for row in rows:
                domains[row["domain"]] = {
                    "count": row["count"],
                    "significances": row["significances"].split(",") if row["significances"] else []
                }

            # Get total count
            total = conn.execute(
                "SELECT COUNT(*) FROM number_mentions WHERE value = ?", (value,)
            ).fetchone()[0]

            return {
                "value": value,
                "total_occurrences": total,
                "domains": domains
            }

    def _row_to_number_mention(self, row: sqlite3.Row) -> NumberMention:
        """Convert a database row to a NumberMention object."""
        return NumberMention(
            id=row["id"],
            value=row["value"],
            context=row["context"] or "",
            mention_type=MentionType(row["mention_type"]) if row["mention_type"] else MentionType.INCIDENTAL,
            significance=row["significance"] or "",
            finding_id=row["finding_id"],
            source_id=row["source_id"],
            domain=row["domain"],
            domains_found_in=json.loads(row["domains_found_in"]) if row["domains_found_in"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # --- Cross-Reference Operations ---

    def save_cross_reference(self, xref: CrossReference) -> None:
        """Save a cross-reference."""
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cross_references
                (id, finding_id_1, finding_id_2, relationship_type, relationship_value,
                 strength, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                xref.id, xref.finding_id_1, xref.finding_id_2,
                xref.relationship_type, xref.relationship_value,
                xref.strength, xref.created_at.isoformat()
            ))

    def get_cross_references(self, finding_id: str) -> list[CrossReference]:
        """Get all cross-references for a finding."""
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT * FROM cross_references
                WHERE finding_id_1 = ? OR finding_id_2 = ?
            """, (finding_id, finding_id)).fetchall()
            return [self._row_to_cross_reference(row) for row in rows]

    def _row_to_cross_reference(self, row: sqlite3.Row) -> CrossReference:
        """Convert a database row to a CrossReference object."""
        return CrossReference(
            id=row["id"],
            finding_id_1=row["finding_id_1"],
            finding_id_2=row["finding_id_2"],
            relationship_type=row["relationship_type"],
            relationship_value=row["relationship_value"] or "",
            strength=row["strength"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # --- Statistics ---

    def get_statistics(self) -> dict:
        """Get database statistics."""
        with self._connection() as conn:
            stats = {}

            stats["sources"] = conn.execute(
                "SELECT COUNT(*) FROM sources"
            ).fetchone()[0]

            stats["trusted_sources"] = conn.execute(
                "SELECT COUNT(*) FROM sources WHERE trust_score >= 70"
            ).fetchone()[0]

            stats["findings"] = conn.execute(
                "SELECT COUNT(*) FROM findings"
            ).fetchone()[0]

            stats["concepts"] = conn.execute(
                "SELECT COUNT(*) FROM concepts"
            ).fetchone()[0]

            stats["number_mentions"] = conn.execute(
                "SELECT COUNT(*) FROM number_mentions"
            ).fetchone()[0]

            stats["cross_references"] = conn.execute(
                "SELECT COUNT(*) FROM cross_references"
            ).fetchone()[0]

            # Most mentioned numbers
            number_rows = conn.execute("""
                SELECT value, COUNT(*) as count
                FROM number_mentions
                GROUP BY value
                ORDER BY count DESC
                LIMIT 10
            """).fetchall()
            stats["top_numbers"] = {row["value"]: row["count"] for row in number_rows}

            return stats
