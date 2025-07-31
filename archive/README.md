# Development Files Archive

This directory contains development files that were moved from the root directory during project sanitization on 2025-07-31.

## Directory Structure

### `development-files/`
Contains legacy development files that are no longer needed in the main project structure but are preserved for reference.

#### `sql-iterations/`
- Multiple iterations of database setup SQL files
- **Files**: `database_setup*.sql`, `create_workspace*.sql`, `*rls*.sql`
- **Reason**: Only the final production database schema is needed in the main project

#### `test-scripts/`
- Standalone test and demo scripts used during development
- **Files**: `demo_pdf_processing.py`, `test_*.py`
- **Reason**: Proper tests are now in the `tests/` directory

#### `server-implementations/`
- Alternative server implementations created during development
- **Files**: `debug_server.py`, `minimal_*.py`, `simple_fastapi.py`, `start_server.py`
- **Reason**: Production server is `web/main.py`

#### `legacy-databases/`
- Old database files that were replaced by the current data management system
- **Files**: `*.db`, `*.sqlite`
- **Reason**: Current database files are managed in `data/` directory

#### `status-docs/`
- Status documentation from specific development phases
- **Files**: `SERPBEAR_FINAL_STATUS.md`, `INITIAL.md`
- **Reason**: Information integrated into main documentation

## Impact of Cleanup

**Before**: ~100 files in root directory
**After**: ~30 core production files in root directory

This cleanup improves:
- Project navigation and understanding
- Development workflow clarity
- Professional appearance
- Reduced confusion for new developers

## Recovery

If any archived file is needed, it can be moved back to the appropriate location. All files were moved using `git mv` to preserve version history.