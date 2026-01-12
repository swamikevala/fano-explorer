"""
Researcher orchestrator - main control loop.

Coordinates observation, question generation, searching, fetching,
extraction, and storage.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import yaml

from shared.logging import get_logger

from .observers import ContextAggregator
from .questions import QuestionGenerator
from .sources import WebSearcher, ContentFetcher, ContentCache
from .sources.trust import TrustEvaluator
from .analysis import ContentExtractor, CrossReferenceDetector
from .store.database import ResearcherDatabase
from .store.models import Source, SourceTier

log = get_logger("researcher", "orchestrator")


class Orchestrator:
    """
    Main orchestrator for the researcher module.

    Runs the continuous research loop:
    1. Observe explorer/documenter activity
    2. Generate research questions
    3. Search for sources
    4. Fetch and evaluate sources
    5. Extract findings
    6. Store and cross-reference
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize orchestrator.

        Args:
            base_path: Base path of the fano project (auto-detected if None)
        """
        # Detect base path
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent.parent
        self.base_path = base_path

        # Paths
        self.researcher_path = base_path / "researcher"
        self.explorer_path = base_path / "explorer"
        self.documenter_path = base_path / "documenter"
        self.config_path = self.researcher_path / "config"
        self.data_path = self.researcher_path / "data"

        # Load config
        self.config = self._load_config()

        # Initialize components
        self._init_components()

        # State
        self._running = False
        self._idle_count = 0
        self._last_search_time: Optional[datetime] = None
        self._questions_processed = 0

    def _load_config(self) -> dict:
        """Load settings configuration."""
        settings_path = self.config_path / "settings.yaml"
        try:
            with open(settings_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            log.warning("orchestrator.config_load_failed", error=str(e))
            return {}

    def _init_components(self):
        """Initialize all orchestrator components."""
        # Database
        self.db = ResearcherDatabase(self.data_path / "researcher.db")

        # Observers
        self.context_aggregator = ContextAggregator(
            explorer_data_path=self.explorer_path / "data",
            documenter_path=self.documenter_path,
            domains_config_path=self.config_path / "domains.yaml"
        )

        # Question generation
        self.question_generator = QuestionGenerator(
            templates_path=self.config_path / "question_templates.yaml",
            domains_path=self.config_path / "domains.yaml"
        )

        # Sources
        self.searcher = WebSearcher(
            config_path=self.config_path / "settings.yaml",
            trusted_sources_path=self.config_path / "trusted_sources.yaml"
        )
        self.fetcher = ContentFetcher(
            config_path=self.config_path / "settings.yaml",
            cache_dir=self.data_path / "cache"
        )
        self.cache = ContentCache(
            cache_dir=self.data_path / "cache",
            config_path=self.config_path / "settings.yaml"
        )

        # Trust evaluation
        self.trust_evaluator = TrustEvaluator(
            trusted_sources_path=self.config_path / "trusted_sources.yaml",
            config_path=self.config_path / "settings.yaml",
            llm_client=None  # Set later if available
        )

        # Analysis
        self.extractor = ContentExtractor(
            config_path=self.config_path / "settings.yaml",
            domains_config_path=self.config_path / "domains.yaml",
            llm_client=None  # Set later if available
        )
        self.cross_ref_detector = CrossReferenceDetector(self.db)

        log.info("orchestrator.initialized")

    def set_llm_client(self, llm_client):
        """
        Set LLM client for smart extraction and trust evaluation.

        Args:
            llm_client: LLM client with async query() method
        """
        self.trust_evaluator.llm_client = llm_client
        self.extractor.llm_client = llm_client
        log.info("orchestrator.llm_client_set")

    async def run(self):
        """
        Run the main research loop.

        Continues until stopped.
        """
        self._running = True
        log.info("orchestrator.started")

        observer_config = self.config.get("observer", {})
        polling_interval = observer_config.get("polling_interval_seconds", 30)
        idle_interval = observer_config.get("idle_polling_interval_seconds", 300)
        idle_threshold = observer_config.get("idle_threshold_checks", 10)

        while self._running:
            try:
                # Update context from observers
                context = self.context_aggregator.update_context()

                # Check if idle
                is_idle = self.context_aggregator.is_idle()
                if is_idle:
                    self._idle_count += 1
                else:
                    self._idle_count = 0

                # Determine sleep interval
                current_interval = (
                    idle_interval if self._idle_count >= idle_threshold
                    else polling_interval
                )

                # Run research cycle if not idle or on schedule
                if not is_idle or self._should_research():
                    await self._research_cycle(context)

                # Log status periodically
                if self._questions_processed % 10 == 0:
                    stats = self.db.get_statistics()
                    log.info(
                        "orchestrator.status",
                        findings=stats.get("findings", 0),
                        sources=stats.get("sources", 0),
                        questions_processed=self._questions_processed,
                        idle_count=self._idle_count
                    )

                # Sleep
                await asyncio.sleep(current_interval)

            except asyncio.CancelledError:
                log.info("orchestrator.cancelled")
                break
            except Exception as e:
                log.error("orchestrator.error", error=str(e))
                await asyncio.sleep(polling_interval)

        log.info("orchestrator.stopped")

    def _should_research(self) -> bool:
        """Determine if we should do research even when idle."""
        if self._last_search_time is None:
            return True

        # Research at least every 10 minutes even when idle
        elapsed = datetime.now() - self._last_search_time
        return elapsed > timedelta(minutes=10)

    async def _research_cycle(self, context):
        """
        Run one research cycle.

        Args:
            context: Current research context
        """
        log.debug("orchestrator.cycle_start")

        # Generate questions from context
        questions = self.question_generator.generate(
            context,
            max_questions=self.config.get("limits", {}).get("max_questions_per_cycle", 20)
        )

        if not questions:
            log.debug("orchestrator.no_questions")
            return

        # Prioritize questions
        questions = self.question_generator.prioritize_by_context(questions, context)

        # Process top questions
        for question in questions[:5]:  # Limit per cycle
            await self._process_question(question, context)
            self._questions_processed += 1

        self._last_search_time = datetime.now()

    async def _process_question(self, question: dict, context):
        """
        Process a single research question.

        Args:
            question: Question dict with query, type, etc.
            context: Research context
        """
        query = question["query"]
        log.debug("orchestrator.processing_question", query=query[:50])

        # Search for sources
        search_results = await self.searcher.search(
            query=query,
            domain=question.get("source_value") if question["source"] == "domain" else None,
            limit=5
        )

        if not search_results:
            return

        # Process each result
        for result in search_results[:3]:  # Limit fetches per question
            await self._process_source(result, context)

    async def _process_source(self, search_result: dict, context):
        """
        Process a single source.

        Args:
            search_result: Search result with URL
            context: Research context
        """
        url = search_result.get("url")
        if not url:
            return

        # Check if we've already processed this URL
        existing = self.db.get_source_by_url(url)
        if existing:
            log.debug("orchestrator.source_exists", url=url[:50])
            return

        # Fetch content
        content = await self.fetcher.fetch(url)
        if not content:
            return

        # Evaluate trust
        trust_result = await self.trust_evaluator.evaluate(
            url=url,
            content=content,
            use_llm=self.trust_evaluator.llm_client is not None
        )

        # Skip low-trust sources
        min_trust = self.config.get("trust", {}).get("min_trust_score", 50)
        if trust_result["trust_score"] < min_trust:
            log.debug(
                "orchestrator.source_low_trust",
                url=url[:50],
                score=trust_result["trust_score"]
            )
            return

        # Create and save source
        source = Source.create(
            url=url,
            domain=content["domain"],
            title=content.get("title", ""),
            content_hash=content["content_hash"],
            content_type=content.get("content_type", "html")
        )
        source.trust_score = trust_result["trust_score"]
        source.trust_tier = trust_result["trust_tier"]
        source.evaluation_reasoning = trust_result["reasoning"]
        source.evaluated_at = datetime.now()
        source.has_sanskrit_citations = trust_result.get("observable_features", {}).get("has_sanskrit", False)
        source.has_verse_references = trust_result.get("observable_features", {}).get("has_verse_references", False)
        source.has_bibliography = trust_result.get("observable_features", {}).get("has_bibliography", False)
        source.is_academic_domain = trust_result.get("observable_features", {}).get("is_academic", False)

        self.db.save_source(source)
        log.info(
            "orchestrator.source_saved",
            url=url[:50],
            trust_score=source.trust_score
        )

        # Update trusted sources file if high trust
        if source.trust_score >= 70:
            self.trust_evaluator.update_trusted_sources(source.domain, trust_result)

        # Extract findings
        extraction_result = await self.extractor.extract(
            source_id=source.id,
            content=content,
            research_context=context.to_dict(),
            use_llm=self.extractor.llm_client is not None
        )

        # Save findings
        for finding in extraction_result.get("findings", []):
            self.db.save_finding(finding)

            # Detect cross-references
            xrefs = self.cross_ref_detector.find_cross_references(
                finding,
                extraction_result.get("numbers", [])
            )
            for xref in xrefs:
                self.db.save_cross_reference(xref)

        # Save number mentions
        for number in extraction_result.get("numbers", []):
            self.db.save_number_mention(number)

        log.info(
            "orchestrator.extraction_complete",
            source_id=source.id,
            findings=len(extraction_result.get("findings", [])),
            numbers=len(extraction_result.get("numbers", []))
        )

    def stop(self):
        """Stop the orchestrator."""
        self._running = False
        log.info("orchestrator.stop_requested")

    def get_status(self) -> dict:
        """Get current orchestrator status."""
        stats = self.db.get_statistics()
        cache_stats = self.cache.get_stats()

        return {
            "running": self._running,
            "questions_processed": self._questions_processed,
            "idle_count": self._idle_count,
            "last_search": self._last_search_time.isoformat() if self._last_search_time else None,
            "database": stats,
            "cache": cache_stats,
        }
