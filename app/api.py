"""Run: python -m uvicorn app.api:api --host 127.0.0.1 --port 8000."""
from typing import Annotated, Literal
from uuid import UUID
import hashlib
import io

import numpy as np
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response, UploadFile, File, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr

from app import database as db
from app.data import case_index, default_root, load_case
from app.reporting import build_pdf
from app.pancreas import ModelUnavailable, ScanTooLarge, model_available, segment_pancreas
from app.lesion import model_available as lesion_model_available, segment_lesion
from app.classification import (NoPancreasRegion, classify_pdac, frozen_config,
                                model_available as classifier_model_available)

api = FastAPI(title='PanVision Research API', version='0.1.0')
bearer = HTTPBearer(auto_error=False)
StudyID = Annotated[str, Path(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')]


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Login(Input):
    email: EmailStr = Field(max_length=254)
    password: SecretStr = Field(min_length=1, max_length=1024)


class Notes(Input):
    notes: str = Field(default='', max_length=10000)


class Doctor(BaseModel):
    id: int
    name: str
    email: str
    role: Literal['doctor','admin'] = 'doctor'


class LoginResult(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    expires_at: float
    doctor: Doctor


@api.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # FastAPI's default validation response may include the rejected password/body.
    return JSONResponse(status_code=422, content={'detail': [
        {'loc': list(e['loc']), 'msg': e['msg'], 'type': e['type']} for e in exc.errors()]})


@api.exception_handler(db.DatabaseConfigurationError)
async def configuration_error(request, exc):
    return JSONResponse(status_code=503, content={'detail': 'Database is not configured.'})


@api.exception_handler(psycopg.Error)
async def database_error(request, exc):
    return JSONResponse(status_code=503, content={'detail': 'Database unavailable. Please try again later.'})


@api.middleware('http')
async def private_responses(request: Request, call_next):
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


def credentials(value: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
    if value is None or len(value.credentials) > 256:
        raise HTTPException(401, 'Sign in required.', headers={'WWW-Authenticate': 'Bearer'})
    return value.credentials


def current_doctor(token: Annotated[str, Depends(credentials)]):
    doctor = db.session_doctor(token)
    if doctor is None:
        raise HTTPException(401, 'Session expired or invalid. Sign in again.', headers={'WWW-Authenticate': 'Bearer'})
    return doctor


def studies():
    cases, _ = case_index(default_root())
    return cases


def study(study_id):
    result = next((case for case in studies() if case['study_id'] == study_id), None)
    if result is None:
        raise HTTPException(404, 'Study not found or not available locally.')
    return result


@api.get('/health')
def health():
    return {'status': 'ok'}


@api.get('/ready')
def ready():
    with db.connect() as connection:
        connection.execute('SELECT 1 FROM doctors LIMIT 1')
    return {'status': 'ready'}


@api.post('/auth/login', response_model=LoginResult)
def login(body: Login):
    doctor = db.authenticate(str(body.email), body.password.get_secret_value())
    if doctor is None:
        raise HTTPException(401, 'Unable to sign in. Check details or try again in five minutes.')
    token, expires = db.new_session(doctor['id'])
    return {'access_token': token, 'expires_at': expires, 'doctor': doctor}


@api.get('/auth/me', response_model=Doctor)
def me(doctor: Annotated[dict, Depends(current_doctor)]):
    return doctor


@api.post('/auth/logout', status_code=204)
def logout(token: Annotated[str, Depends(credentials)]):
    db.revoke_session(token)
    return Response(status_code=204)


@api.get('/cases')
def list_cases(doctor: Annotated[dict, Depends(current_doctor)]):
    # Return identifiers and metadata, never server filesystem paths.
    return [{k: c.get(k, '') for k in ('study_id','patient_id','diagnosis','split','annotation_tier')} for c in studies()]


@api.get('/me/cases')
def my_cases(doctor: Annotated[dict, Depends(current_doctor)]):
    return db.saved_cases(doctor['id'])


@api.put('/cases/{study_id}/notes', response_model=Notes)
def update_notes(study_id: StudyID, body: Notes, doctor: Annotated[dict, Depends(current_doctor)]):
    study(study_id)
    db.save_case(doctor['id'], study_id, body.notes)
    return body


@api.post('/cases/{study_id}/report.pdf')
def pdf_report(study_id: StudyID, body: Notes, doctor: Annotated[dict, Depends(current_doctor)]):
    case = study(study_id)
    try:
        array, labels, spacing = load_case(case['processed_image'], case.get('processed_labels', ''))
        pdf = build_pdf(case, array, labels, spacing, body.notes)
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(409, 'Study files are unavailable or invalid. Refresh the dataset and retry.')
    return Response(pdf, media_type='application/pdf',
                    headers={'Content-Disposition': f'attachment; filename="{study_id}_report.pdf"'})


async def read_upload(file):
    import tempfile
    from pathlib import Path as FilePath
    from starlette.concurrency import run_in_threadpool
    from app.validation import MAX_UPLOAD_BYTES, read_ct
    name = (file.filename or '').lower()
    suffix = '.nii.gz' if name.endswith('.nii.gz') else FilePath(name).suffix
    if suffix not in ('.nii', '.nii.gz', '.mha'):
        raise HTTPException(422, 'Upload a .nii, .nii.gz, or .mha CT volume.')
    try:
        with tempfile.TemporaryDirectory(prefix='panvision-api-') as folder:
            path = FilePath(folder) / ('scan' + suffix)
            total = 0
            digest = hashlib.sha256()
            with path.open('wb') as output:
                while chunk := await file.read(1024*1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, 'Upload exceeds 512 MiB.')
                    output.write(chunk)
                    digest.update(chunk)
            try:
                array, spacing = await run_in_threadpool(read_ct, path)
            except (OSError, ValueError, RuntimeError):
                raise HTTPException(422, 'Unreadable or invalid CT. Use a finite 3D NIfTI or self-contained MHA volume.')
            return array, spacing, {'sha256': digest.hexdigest(), 'size_bytes': total}
    finally:
        await file.close()


async def tracked_upload(file, doctor):
    filename = (file.filename or 'unnamed').replace('\\','/').rsplit('/',1)[-1][:255]
    try:
        array, spacing, metadata = await read_upload(file)
    except HTTPException:
        await run_in_threadpool(db.record_upload, doctor['id'], filename, 'rejected')
        raise
    upload_id = await run_in_threadpool(db.record_upload, doctor['id'], filename, 'validated', **metadata,
                                       shape=list(array.shape), spacing=list(spacing))
    return array, spacing, metadata, upload_id


@api.post('/uploads/validate')
async def validate_upload(doctor: Annotated[dict, Depends(current_doctor)], file: Annotated[UploadFile, File()]):
    array, spacing, metadata, upload_id = await tracked_upload(file, doctor)
    return {'valid': True, 'upload_id': upload_id, 'shape_zyx': list(array.shape),
            'spacing_xyz_mm': list(spacing), 'inference_available': model_available(),
            'lesion_inference_available': model_available() and lesion_model_available()}


@api.post('/uploads/{upload_id}/pancreas')
async def pancreas_upload(upload_id: UUID, doctor: Annotated[dict, Depends(current_doctor)],
                          file: Annotated[UploadFile, File()]):
    """Analyze only the doctor's previously validated, byte-matching upload."""
    try:
        record = await run_in_threadpool(db.get_upload, upload_id, doctor['id'])
    except LookupError:
        raise HTTPException(404, 'Upload not found.')
    array, spacing, metadata = await read_upload(file)
    if record['status'] != 'validated' or record['sha256'] != metadata['sha256']:
        raise HTTPException(409, 'Uploaded file does not match the selected history record.')
    try:
        mask = await run_in_threadpool(segment_pancreas, array, spacing)
    except ScanTooLarge:
        raise HTTPException(413, 'Scan is too large for local pancreas segmentation.')
    except ModelUnavailable:
        raise HTTPException(503, 'Pancreas model is not configured on the backend.')
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(503, 'Pancreas segmentation could not complete for this scan.')
    output = io.BytesIO()
    np.savez_compressed(output, mask=mask.astype(np.uint8))
    return Response(output.getvalue(), media_type='application/octet-stream',
                    headers={'X-PanVision-Model': 'P3_nnunet_2d_full/fold_0/checkpoint_best'})


@api.post('/uploads/{upload_id}/lesion')
async def lesion_upload(upload_id: UUID, doctor: Annotated[dict, Depends(current_doctor)],
                        file: Annotated[UploadFile, File()]):
    """Run P3 then L2 on the doctor's byte-matching validated upload."""
    try:
        record = await run_in_threadpool(db.get_upload, upload_id, doctor['id'])
    except LookupError:
        raise HTTPException(404, 'Upload not found.')
    array, spacing, metadata = await read_upload(file)
    if record['status'] != 'validated' or record['sha256'] != metadata['sha256']:
        raise HTTPException(409, 'Uploaded file does not match the selected history record.')
    if not model_available() or not lesion_model_available():
        raise HTTPException(503, 'Pancreas or lesion model is not configured on the backend.')
    try:
        pancreas_mask = await run_in_threadpool(segment_pancreas, array, spacing)
        lesion_mask = await run_in_threadpool(segment_lesion, array, spacing, pancreas_mask)
    except ScanTooLarge:
        raise HTTPException(413, 'Scan is too large for local segmentation.')
    except ModelUnavailable:
        raise HTTPException(503, 'Pancreas or lesion model is not configured on the backend.')
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(503, 'Lesion segmentation could not complete for this scan.')
    output = io.BytesIO()
    np.savez_compressed(output, pancreas=pancreas_mask.astype(np.uint8),
                        lesion=lesion_mask.astype(np.uint8))
    return Response(output.getvalue(), media_type='application/octet-stream',
                    headers={'X-PanVision-Models': 'P3_nnunet_2d_full,L2_attention_unet_roi_full'})


@api.post('/uploads/{upload_id}/classification')
async def classification_upload(upload_id: UUID, doctor: Annotated[dict, Depends(current_doctor)],
                                file: Annotated[UploadFile, File()]):
    """Run P3 then the frozen PDAC classifier on the doctor's byte-matching validated upload."""
    try:
        record = await run_in_threadpool(db.get_upload, upload_id, doctor['id'])
    except LookupError:
        raise HTTPException(404, 'Upload not found.')
    array, spacing, metadata = await read_upload(file)
    if record['status'] != 'validated' or record['sha256'] != metadata['sha256']:
        raise HTTPException(409, 'Uploaded file does not match the selected history record.')
    if not model_available() or not classifier_model_available():
        raise HTTPException(503, 'Pancreas or classification model is not configured on the backend.')
    try:
        pancreas_mask = await run_in_threadpool(segment_pancreas, array, spacing)
        result = await run_in_threadpool(classify_pdac, array, spacing, pancreas_mask)
    except ScanTooLarge:
        raise HTTPException(413, 'Scan is too large for local classification.')
    except NoPancreasRegion:
        raise HTTPException(422, 'The pancreas model found no clear pancreas region, so classification cannot run.')
    except ModelUnavailable:
        raise HTTPException(503, 'Pancreas or classification model is not configured on the backend.')
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(503, 'PDAC classification could not complete for this scan.')
    output = io.BytesIO()
    np.savez_compressed(output, pancreas=pancreas_mask.astype(np.uint8),
                        probability=result['probability'], threshold=result['threshold'],
                        slices=result['slices'])
    return Response(output.getvalue(), media_type='application/octet-stream',
                    headers={'X-PanVision-Models': 'P3_nnunet_2d_full,densenet121_predicted'})


@api.post('/uploads/report.pdf')
async def upload_report(doctor: Annotated[dict, Depends(current_doctor)], file: Annotated[UploadFile, File()],
                        notes: Annotated[str, Form(max_length=10000)] = '',
                        display_mode: Annotated[Literal['auto', 'normalized', 'hu'], Form()] = 'auto',
                        upload_id: Annotated[UUID | None, Form()] = None,
                        pancreas_mask: Annotated[UploadFile | None, File()] = None,
                        pdac_probability: Annotated[float | None, Form(ge=0, le=1)] = None):
    from starlette.concurrency import run_in_threadpool
    recorded_upload = upload_id is not None
    if pancreas_mask is not None and not recorded_upload:
        raise HTTPException(422, 'A validated upload is required for a pancreas overlay.')
    if pdac_probability is not None and not recorded_upload:
        raise HTTPException(422, 'A validated upload is required for a classification result.')
    classification = None
    if pdac_probability is not None:
        try:
            threshold = frozen_config()['threshold']
        except (ModelUnavailable, OSError, ValueError, KeyError):
            raise HTTPException(503, 'Classification model is not configured on the backend.')
        classification = {'probability': pdac_probability, 'threshold': threshold}
    if upload_id is None:
        array, spacing, metadata, upload_id = await tracked_upload(file, doctor)
    else:
        try:
            record = await run_in_threadpool(db.get_upload, upload_id, doctor['id'])
        except LookupError:
            raise HTTPException(404, 'Upload not found.')
        array, spacing, metadata = await read_upload(file)
        if record['sha256'] != metadata['sha256'] or record['status'] != 'validated':
            raise HTTPException(409, 'Uploaded file does not match the selected history record.')
    mask = None
    if pancreas_mask is not None:
        expected = (array.size + 7) // 8
        packed = await pancreas_mask.read(expected + 1)
        if len(packed) != expected or await pancreas_mask.read(1):
            raise HTTPException(422, 'Pancreas mask size does not match the uploaded scan.')
        mask = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=array.size).reshape(array.shape).astype(bool)
    case = {'study_id': str(upload_id), 'source': 'upload', 'diagnosis': 'Unavailable'}
    pdf = await run_in_threadpool(build_pdf, case, array, None, spacing, notes,
                                  display_mode=display_mode, pancreas_mask=mask,
                                  classification=classification)
    try:
        await run_in_threadpool(db.record_report, upload_id, doctor['id'], metadata['sha256'], notes)
    except LookupError:
        raise HTTPException(404, 'Upload not found.')
    return Response(pdf, media_type='application/pdf',
                    headers={'Content-Disposition': 'attachment; filename="panvision-ct-report.pdf"'})


class NewDoctor(Input):
    email: EmailStr = Field(max_length=254)
    name: str = Field(min_length=1, max_length=100, pattern=r'.*\S.*')
    password: SecretStr = Field(min_length=12, max_length=1024)


class AccountUpdate(Input):
    active: bool
    role: Literal['doctor','admin']


def current_admin(doctor: Annotated[dict, Depends(current_doctor)]):
    if doctor.get('role') != 'admin':
        raise HTTPException(403, 'Administrator access required.')
    return doctor


@api.get('/admin/accounts')
def accounts(admin: Annotated[dict, Depends(current_admin)]):
    return db.list_accounts()


@api.post('/admin/doctors', status_code=201)
def create_account(body: NewDoctor, admin: Annotated[dict, Depends(current_admin)]):
    try:
        return db.admin_create_doctor(admin['id'], str(body.email), body.name, body.password.get_secret_value())
    except psycopg.errors.UniqueViolation:
        raise HTTPException(409, 'An account with that email already exists.')
    except PermissionError:
        raise HTTPException(403, 'Administrator access required.')


@api.patch('/admin/accounts/{account_id}')
def update_account(account_id: Annotated[int, Path(gt=0)], body: AccountUpdate,
                   admin: Annotated[dict, Depends(current_admin)]):
    try:
        return db.admin_update_account(admin['id'], account_id, body.active, body.role)
    except PermissionError:
        raise HTTPException(403, 'Administrator access required.')
    except LookupError:
        raise HTTPException(404, 'Account not found.')
    except db.AccountConflict as exc:
        raise HTTPException(409, str(exc))


@api.get('/admin/database')
def database_status(admin: Annotated[dict, Depends(current_admin)]):
    return db.admin_database_status()


class PasswordChange(Input):
    current_password: SecretStr = Field(min_length=1, max_length=1024)
    new_password: SecretStr = Field(min_length=12, max_length=1024)


@api.post('/auth/password', status_code=204)
def change_own_password(body: PasswordChange, doctor: Annotated[dict, Depends(current_doctor)]):
    try:
        changed = db.change_password(doctor['id'], body.current_password.get_secret_value(), body.new_password.get_secret_value())
    except ValueError:
        raise HTTPException(400, 'Choose a different password of at least 12 characters.')
    if not changed:
        raise HTTPException(400, 'Current password is incorrect or the account is temporarily locked.')
    return Response(status_code=204)


@api.get('/me/uploads')
def my_history(doctor: Annotated[dict, Depends(current_doctor)],
               limit: Annotated[int, Query(ge=1,le=100)] = 50, offset: Annotated[int, Query(ge=0)] = 0):
    return db.upload_history(doctor['id'], limit, offset)


@api.get('/admin/uploads')
def all_history(admin: Annotated[dict, Depends(current_admin)], doctor_id: Annotated[int | None, Query(gt=0)] = None,
                limit: Annotated[int, Query(ge=1,le=100)] = 50, offset: Annotated[int, Query(ge=0)] = 0):
    return db.upload_history(doctor_id, limit, offset)


@api.get('/uploads/{upload_id}')
def upload_detail(upload_id: UUID, doctor: Annotated[dict, Depends(current_doctor)]):
    try:
        record = db.get_upload(upload_id, None if doctor.get('role') == 'admin' else doctor['id'])
    except LookupError:
        raise HTTPException(404, 'Upload not found.')
    if record['doctor_id'] != doctor['id']:
        record.pop('notes', None)
    return record


@api.put('/uploads/{upload_id}/notes', response_model=Notes)
def upload_notes(upload_id: UUID, body: Notes, doctor: Annotated[dict, Depends(current_doctor)]):
    try:
        db.save_upload_notes(upload_id, doctor['id'], body.notes)
    except LookupError:
        raise HTTPException(404, 'Upload not found.')
    return body


@api.get('/auth/session', response_model=Doctor)
def session_status(token: Annotated[str, Depends(credentials)]):
    doctor = db.session_doctor(token, touch=False)
    if doctor is None:
        raise HTTPException(401, 'Session expired or invalid. Sign in again.', headers={'WWW-Authenticate':'Bearer'})
    return doctor
