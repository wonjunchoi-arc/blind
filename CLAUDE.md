# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
- **Activate virtual environment**: `.\blindinsight-env\Scripts\Activate.ps1` (Windows PowerShell) or `blindinsight-env\Scripts\activate.bat` (Command Prompt)
- **Install dependencies**: `uv pip install -r requirements.txt` (preferred) or `pip install -r requirements.txt`
- **Environment check**: `python main.py --check-requirements` (validates all required packages)

### Running the Application
- **Start application**: `python main.py` or `streamlit run main.py`
- **Development mode**: Set `enable_debug_mode = True` in src/blindinsight/models/base.py Settings class
- **Access URL**: http://localhost:8501

### Testing & Quality
- **Run tests**: `pytest tests/` (unit tests in tests/unit/, integration tests in tests/integration/)
- **Test markers**: Use `-m "not slow"` to skip slow tests, `-m integration` for integration tests only
- **Code formatting**: `black src/` (line length: 88)
- **Type checking**: `mypy src/` (strict typing enforced)
- **Linting**: `flake8 src/`
- **Coverage**: `coverage run -m pytest` then `coverage report`

### Development Workflow
- **Single test**: `pytest tests/unit/test_specific_module.py::test_function_name`
- **Watch mode**: `pytest tests/ --watch` (requires pytest-watch)
- **Debug mode**: Set breakpoints and run `python -m pytest tests/ -s --pdb`

## High-Level Architecture

### Core System Design
BlindInsight AI is a **LangGraph-based multi-agent orchestration system** that combines RAG (Retrieval Augmented Generation) with MCP (Model Context Protocol) for comprehensive company analysis.

#### LangGraph StateGraph Workflow
The system uses LangGraph's StateGraph to orchestrate multiple AI agents:
- **Sequential Mode**: `input_validation → culture_analysis → compensation_analysis → growth_analysis → career_analysis → synthesis → report_generation`
- **Parallel Mode**: `input_validation → [culture|compensation|growth in parallel] → career_analysis → synthesis → report_generation`

Control via `WorkflowConfig.enable_parallel_execution` in src/blindinsight/workflow/state.py

#### Agent Hierarchy
All agents inherit from `BaseAgent` (src/blindinsight/agents/base.py) which provides:
- RAG knowledge retrieval via `retrieve_knowledge()`
- LLM response generation via `generate_response()`
- Result validation and performance tracking
- Configurable search parameters (k, threshold, reranking)

Specialized agents (src/blindinsight/agents/):
- `SalaryBenefitsAgent`: Salary and benefits analysis (salary_benefits_agent.py)
- `CompanyCultureAgent`: Company culture analysis (company_culture_agent.py)
- `WorkLifeBalanceAgent`: Work-life balance analysis (work_life_balance_agent.py)
- `ManagementAgent`: Management and leadership analysis (management_agent.py)
- `CareerGrowthAgent`: Career growth opportunities (career_growth_agent.py)
- `QualityEvaluatorAgent`: Analysis quality evaluation (quality_evaluator_agent.py)

### Data Architecture

#### RAG System (src/blindinsight/rag/)
- **Vector Database**: ChromaDB with collections: company_culture, work_life_balance, management, salary_benefits, career_growth, general
- **Embeddings**: OpenAI text-embedding-3-small (default, configurable via EMBEDDING_MODEL env var)
- **Hybrid Retrieval**: BM25 + Vector search ensemble with configurable weights
- **Reranking**: Optional reranking for improved relevance (controlled by rerank parameter)
- **Batch Processing**: BatchEmbeddingManager for optimized embedding generation

#### Supervisor Chat System (src/blindinsight/chat/)
- **ModernSupervisorAgent**: LangGraph-based multi-agent orchestration
- **SupervisorState**: Conversation state management with message history
- **Command Handoffs**: Tool-based agent delegation system
- **Specialized Agents**: 5 domain experts + quality evaluator coordinated by supervisor
- **Workflow**: LangGraph StateGraph with conditional routing and human-in-the-loop support

#### State Management
- **SupervisorState** (src/blindinsight/chat/state.py): Manages conversation flow, message history, and agent handoffs
- **Configuration**: Model selection, search parameters, and agent routing via environment variables

### Configuration System
Settings managed through `Settings` dataclass (src/blindinsight/models/base.py):
- Environment variables auto-loaded from .env file
- API keys: OPENAI_API_KEY (required), ANTHROPIC_API_KEY (optional)
- Validation enforces directory creation and value ranges
- Global settings instance accessible via `from blindinsight.models.base import settings`

## Key Implementation Patterns

### Adding New Agents
To add new specialized agents:
1. Create new agent class inheriting from `BaseAgent` in src/blindinsight/agents/
2. Implement agent-specific prompt and category mapping
3. Register in `SupervisorState` and handoff tools (src/blindinsight/chat/workflow.py)
4. Add corresponding environment variable for model selection (optional)
5. Update ChromaDB collection mappings if using new categories

### Error Handling & Performance
- All agents use `_execute_with_error_handling()` wrapper
- Performance metrics tracked per agent execution
- Comprehensive validation via `validate_result()`
- Retry logic and timeout handling built into base classes

### Async Architecture
- LangGraph workflow execution is fully async
- Agent orchestration supports both sequential and parallel execution
- MCP service calls are async-compatible
- Use `asyncio.gather()` for parallel agent execution

### Environment Requirements
- Python 3.10+ (specified in pyproject.toml)
- OpenAI API key required for LLM operations
- All dependencies pinned in requirements.txt
- Development tools: black, mypy, pytest configured in pyproject.toml

## Data Collection & Processing

### Crawling Blind Reviews
For detailed crawling documentation, see [tools/README.md](tools/README.md)

**Quick Start**:
```bash
# Interactive crawling
python tools/blind_review_crawler.py

# Single company (AI classification)
python -c "
from tools.blind_review_crawler import run_single_company_crawl
run_single_company_crawl('NAVER', pages=25, use_ai_classification=True)
"
```

**Key Features**: AI classification, keyword fallback, 90% API cost savings via batch processing

### Data Vectorization & Migration
```bash
# Full data migration
python migrate_reviews.py

# Specific company only
python migrate_reviews.py NAVER

# Help
python migrate_reviews.py --help
```

**Migration Pipeline**:
1. Load chunk data from JSON files
2. Generate embeddings using OpenAI model (batch optimized)
3. Store in ChromaDB by category collections
4. Validate data integrity and generate performance report

**Key Files**:
- **Crawler**: tools/blind_review_crawler.py (Selenium-based)
- **Category Processor**: tools/enhanced_category_processor.py (AI classification)
- **Migration**: migrate_reviews.py (batch embedding generation)
- **RAG Retriever**: src/blindinsight/rag/retriever.py (hybrid search)

## Critical File Locations
- **Main entry**: main.py → src/blindinsight/frontend/app.py
- **Supervisor Chat**: src/blindinsight/chat/modern_supervisor.py (multi-agent orchestration)
- **Agent base classes**: src/blindinsight/agents/base.py
- **Configuration**: src/blindinsight/models/base.py
- **State management**: src/blindinsight/chat/state.py (SupervisorState)
- **Frontend interface**: src/blindinsight/frontend/app.py
- **Data crawler**: tools/blind_review_crawler.py
- **Data migration**: migrate_reviews.py