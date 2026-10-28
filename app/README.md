# PanVision UI and API

PanVision is a research workspace for reviewing pancreatic CT scans. It combines
a green-and-cream Streamlit dashboard, a FastAPI backend, and PostgreSQL account,
upload-history and audit tables. It is an academic prototype, not a clinical
system. P3 pancreas segmentation can run when its model files and optional
inference dependencies are installed. Lesion segmentation, classification,
Grad-CAM and retrieval are not connected; the interface does not invent their results.

## Folder structure

```text
app/
├── app.py                 # Streamlit entry point
├── dashboard.py           # Doctor/admin pages and upload/report workflow
├── auth.py                # Login screen and non-renewing idle-session monitor
├── api_client.py          # Server-side Streamlit → FastAPI HTTP requests
├── api.py                 # API routes, request validation and authorization
├── database.py            # PostgreSQL schema, accounts, sessions, history and CLI
├── validation.py          # Bounded CT file/geometry/intensity validation
├── data.py                # Image display helpers and optional research-case reader
├── reporting.py           # PDF generation and research HTML report helper
├── style.css              # Dashboard layout and green/cream appearance
├── requirements.txt       # Frontend/backend dependencies
├── __init__.py            # Python package marker
└── tests/
    ├── test_api.py        # API validation, authentication and ownership checks
    ├── test_data.py       # CT display and research-case indexing
    ├── test_database.py   # Configuration plus optional PostgreSQL integration tests
    ├── test_session_ui.py # Idle sign-out clears private session state
    └── test_ui.py         # Login, navigation, upload, report, profile/admin workflows
```

The repository's `.streamlit/config.toml` sets the light theme, localhost bind and
512 MiB upload limit. `.gitignore` excludes local secrets, the virtual environment,
bytecode, output files, scans and weights. The preprocessing code and shared
patient split remain in `src/preprocessing/` and `data/splits.csv`; this UI does
not alter or regenerate them.

## Architecture and workflow

```text
Browser → Streamlit (8501) → FastAPI (8001) → PostgreSQL
                 └── temporary CT preview     ├── accounts and hashed sessions
                                             ├── upload metadata and saved notes
                                             └── upload/account audit events
```

1. An administrator provisions doctor accounts; there is no public registration.
2. A doctor signs in and sees Home with empty imaging/result panels.
3. Upload CT → Validate and open report sends the scan to FastAPI for validation.
4. The API records uploader identity and scan metadata. Streamlit keeps a temporary
   local preview of the validated file and opens Case Report.
5. The doctor reviews slices, saves notes, and prepares/downloads a PDF.
6. PostgreSQL retains upload history, notes and events across logins. CT bytes and
   generated PDF bytes are not stored in PostgreSQL or retained as shared files.

Only FastAPI needs the database URL. Streamlit uses `PANVISION_API_URL` and keeps
its bearer token in server-side session state, not browser query parameters.
The upload-first workflow needs no OneDrive dataset directory.

## Screens

| Screen | Behavior |
| --- | --- |
| Home | Scan metadata strip, CT preview, model-output tabs, prediction/measurement/reference cards, findings and actions |
| Upload CT | NIfTI/MHA uploader, validation and automatic navigation to the report |
| Case Report | Uploaded CT, optional P3 pancreas mask overlay, slice/display controls, geometry, saved notes and PDF generation/download |
| Past Cases | Signed-in doctor's persistent own-upload history, validation status and event timeline |
| Similar Cases | Explicit unavailable state until trained retrieval is connected |
| My Profile | Own name/email/role plus current-password-verified password changes for both roles |
| Administration | Admin-only account creation/access control, uploader-filtered history, database counts and audit events |

History supports 50-row pagination, current-page filename/uploader search and
validation filtering, plus CSV export. Times are shown in UTC. A `pdf_generated`
event means the server generated a report, not proof of a browser download.
Administrators can view upload metadata/events but not another doctor's private
review notes. History begins with uploads made after its schema was installed.
Re-upload the original CT to view images after signing out.

## Local setup

Run commands from the repository root. Python 3.12 is the version used during
development. Create `.venv` only if it does not already exist:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r app/requirements.txt
```

On Windows use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.
Use the explicit virtual-environment interpreter to avoid accidentally running
Streamlit through Conda or a global Python installation.

### 1. Configure Neon privately

In your Neon project, choose the correct branch and database, click **Connect**,
and copy its PostgreSQL connection string. Copy `.env.example` to `.env` in the
repository root, then replace the placeholder URL with the complete Neon URL:

```bash
cp .env.example .env
```

The backend loads the root `.env` automatically. An environment variable already
set in the backend process takes precedence over the file. Keep Neon's TLS query
parameters; do not print, share or commit the URL. `.env` is ignored by Git.
GitHub repository secrets are available only to workflows or deployments that
explicitly pass them to the process; they do not configure a local run.

### 2. Initialize tables and create the first administrator

From the repository root, after configuring `.env`:

```bash
.venv/bin/python -m app.database check
.venv/bin/python -m app.database init
.venv/bin/python -m app.database create-admin
```

Run `init` only after checking that this is the intended Neon database. It is
additive/idempotent. If your existing accounts were migrated from Azure, skip
`create-admin` and sign in with an existing administrator account. Changing the
connection URL alone does not copy Azure data.

Administrator email/name/password are prompted; passwords are hidden. Use a new
email if a doctor account already uses the address. There are no default
credentials. The app administrator is distinct from the PostgreSQL provider
administrator. Existing doctors keep their password hashes and default to doctor
access. `init` is additive/idempotent: it creates missing application tables and
columns without dropping existing data. It requires schema creation privileges;
use appropriately separated admin/runtime database roles when deploying.

### 3. Start FastAPI

Keep the configured backend terminal open:

```bash
.venv/bin/python -m uvicorn app.api:api --host 127.0.0.1 --port 8001
```

API docs: <http://127.0.0.1:8001/docs>. `/health` checks the process; `/ready` checks
the database and basic schema. Stop an older process with Ctrl+C before reusing
its port. Restart the backend after code or schema changes.

### 4. Start Streamlit in another terminal

```bash
export PANVISION_API_URL='http://127.0.0.1:8001'
.venv/bin/python -m streamlit run app/app.py
```

Open <http://127.0.0.1:8501>. The API URL defaults to port 8000 if not configured,
so set it explicitly when using 8001. Sign in as the administrator and add doctors
through Administration → Add doctor. A private `create-doctor` maintenance CLI
also exists for operators with database credentials; it is not public registration.

### Optional: run P3 pancreas segmentation

Install the inference dependencies in the same virtual environment:

```bash
.venv/bin/python -m pip install -r app/inference-requirements.txt
```

Copy the **full** P3 nnU-Net trainer folder into the Git-ignored location below.
It needs `plans.json`, `dataset.json`, and `fold_0/checkpoint_best.pth`:

```text
data/private/models/P3_nnunet_2d_full/nnUNetTrainer_PanVision__nnUNetPlans__2d/
```

Alternatively, set `PANVISION_PANCREAS_MODEL_DIR` in the root `.env` to the
absolute path of that trainer folder. Restart FastAPI after changing model files
or configuration. After a scan passes validation, pancreas segmentation runs
automatically before the case report opens. If it fails, the validated scan
remains available and the report shows a **Retry pancreas segmentation** button.
The backend
uses the training preprocessing (LPS, 1.5 × 1.5 × 3 mm, −100 to 300 HU), runs fold 0's
best checkpoint, and aligns the predicted mask with the uploaded CT. The green
overlay is a research estimate, not a diagnosis. Local inference accepts scans
up to 32 million voxels and may take several minutes on a CPU. The mask lives
only in the current Streamlit session; the PDF still
contains the CT preview and notes, without the mask. Persistent results belong
to the later case-storage work.

## PostgreSQL tables

| Table | Contents |
| --- | --- |
| `doctors` | Name/email, role, salted scrypt password hash, active status and failed-login lockout |
| `doctor_sessions` | SHA-256 token hashes, expiry and last activity; never plaintext tokens |
| `scan_uploads` | Uploader ID, filename, file digest/size, geometry, validation status, notes and timestamps |
| `scan_events` | Validation/rejection, note-save and PDF-generation events |
| `admin_audit` | Account-access and password-change metadata, without passwords or report content |
| `doctor_cases` | Doctor-owned notes for the optional research-dataset API |

Account changes are audited transactionally. Disabling an account or changing its
role revokes its sessions. The last active administrator cannot be disabled or
demoted. Only admins can create doctors through the API, and the backend enforces
that rule independently of the sidebar. Notes are owner-scoped. Linking a PDF
to an upload ID requires matching ownership and file digest.

## Sessions, password changes and validation

- Both roles have a **20-minute sliding inactivity timeout**. Normal authenticated
  app/API actions renew it; the UI's 15-second `/auth/session` checks do not.
- Expiry clears credentials, temporary scans, drafts and prepared PDFs and returns
  to login. Browser suspension may delay the visible redirect, but the backend
  still rejects expired tokens. Reading a scan or typing into an unsubmitted form
  does not count as an authenticated action; save notes before long pauses.
- Password changes require the current password and a different new password of
  12–1024 characters. Success revokes all sessions, including the current one.
- Five failed password attempts lock the account for five minutes. Forgotten-
  password recovery and name/email edits are not implemented.
- Request models reject extra fields, invalid emails/IDs, and notes over 10,000
  characters. Errors omit submitted passwords and raw database exception details.
- CT validation accepts finite 3D `.nii`, `.nii.gz` or self-contained `.mha`, up
  to 512 MiB and 128 million voxels. External-file MHA references are rejected.
- Automatic display uses normalized 0–1 values when detected, otherwise a CT HU
  window. A manual override is available and carried into the PDF. This heuristic
  does not establish acquisition calibration or diagnostic suitability.

CT data, masks and model weights belong in separately configured private file
storage, not Git or PostgreSQL. Upload metadata/notes are persistent; CT files
remain temporary. Lesion/classification inference and persistent scan/PDF storage are future
integration work. Deployments still need HTTPS, trusted PostgreSQL TLS,
proxy upload-size limits, login/upload rate limiting, backups, retention controls
and appropriate authorization before handling identifiable patient data.

## API overview

| Route | Access / purpose |
| --- | --- |
| `GET /health`, `GET /ready` | Process and database readiness |
| `POST /auth/login`, `POST /auth/logout` | Sign in / revoke session |
| `GET /auth/me`, `GET /auth/session` | Own profile; non-renewing session check |
| `POST /auth/password` | Change own password and revoke sessions |
| `POST /uploads/validate` | Validate multipart CT and return its recorded upload ID |
| `POST /uploads/{id}/pancreas` | Run P3 pancreas segmentation for a doctor's byte-matching validated upload |
| `POST /uploads/report.pdf` | Generate PDF from CT, notes, display mode and optional recorded upload ID |
| `GET /me/uploads` | Own paginated upload history |
| `GET /uploads/{id}`, `PUT /uploads/{id}/notes` | Authorized detail and owner-only notes |
| `GET /admin/accounts`, `POST /admin/doctors` | Admin-only account listing/creation |
| `PATCH /admin/accounts/{id}` | Admin-only account access/role update |
| `GET /admin/uploads`, `GET /admin/database` | Admin-only uploader-filtered history and database/audit overview |

Except health/readiness and login, these routes require a bearer token. Optional
research-case routes (`/cases`, `/me/cases`, case notes/PDF) remain available but
are not used to populate the doctor's upload-first dashboard. They read
`PANVISION_DATA_DIR` or ignored `data/storage.json`. CORS is not enabled because
Streamlit calls FastAPI server-side; remote API URLs must use HTTPS.

## Tests

```bash
.venv/bin/python -m unittest discover -s app/tests
```

API/UI tests use synthetic scans and mocked database operations. Real PostgreSQL
integration tests require `PANVISION_TEST_DATABASE_URL` pointing to a disposable
database where the test user can create schemas. Tests create/remove isolated
schemas. Without the variable they are explicitly skipped, not counted as verified
live database behavior. No test should require a teammate's OneDrive files.

## Optional SQLite import

For an older local prototype only: back up SQLite and point the connection URL at
an empty PostgreSQL database, then run:

```bash
.venv/bin/python -m app.database migrate-sqlite --source data/private/panvision.sqlite3
```

The import preserves IDs, hashes, notes and timestamps, refuses a populated target,
and opens the SQLite source read-only. Normal operation uses PostgreSQL only.
