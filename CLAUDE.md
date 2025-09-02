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
- MCP service integration via `call_mcp_service()`
- LLM response generation via `generate_response()`
- Result validation and performance tracking

Specialized agents:
- `CultureAnalysisAgent`: Company culture and work-life balance analysis
- `CompensationAnalysisAgent`: Salary and benefits analysis  
- `GrowthStabilityAgent`: Growth potential and financial stability
- `CareerPathAgent`: Personalized career path recommendations

### Data Architecture

#### RAG System (src/blindinsight/rag/)
- **Vector Database**: ChromaDB with collections: culture_reviews, salary_discussions, career_advice, interview_reviews, company_general
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **Retrieval**: Configurable similarity threshold (default: 0.7), max 20 documents per query

#### MCP Integration (src/blindinsight/mcp/)
External data source providers:
- `BlindDataProvider`: Blind.com review data
- `JobSiteProvider`: Job posting information
- `SalaryDataProvider`: Compensation benchmarks
- `CompanyNewsProvider`: News and trend analysis

#### State Management
`WorkflowState` (src/blindinsight/workflow/state.py) maintains:
- Analysis request and user profile
- Progress tracking and error handling
- Agent execution results
- Performance metrics and timing

### Configuration System
Settings managed through `Settings` dataclass (src/blindinsight/models/base.py):
- Environment variables auto-loaded from .env file
- API keys: OPENAI_API_KEY (required), ANTHROPIC_API_KEY (optional)
- Validation enforces directory creation and value ranges
- Global settings instance accessible via `from blindinsight.models.base import settings`

## Key Implementation Patterns

### Workflow Customization
To add new analysis types:
1. Create new agent class inheriting from `BaseAgent`
2. Add node wrapper in `BlindInsightWorkflow._build_workflow()`
3. Update edge definitions in `_define_workflow_edges()`
4. Register in agent factory functions

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

## Critical File Locations
- **Main entry**: main.py → src/blindinsight/frontend/app.py
- **Workflow orchestration**: src/blindinsight/workflow/graph.py
- **Agent base classes**: src/blindinsight/agents/base.py
- **Configuration**: src/blindinsight/models/base.py
- **State management**: src/blindinsight/workflow/state.py
- **Frontend interface**: src/blindinsight/frontend/app.py