"""Server-side Streamlit client. Credentials never go into browser URLs."""
import os
from urllib.parse import urlsplit
import httpx
import streamlit as st


class APIError(RuntimeError):
    pass


class SessionExpired(APIError):
    pass


def request(method, path, *, authenticated=True, timeout=60, **kwargs):
    base = os.environ.get('PANVISION_API_URL', 'http://127.0.0.1:8000').rstrip('/')
    parsed = urlsplit(base)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise APIError('Configure a valid PANVISION_API_URL without credentials.')
    internal = {host.strip() for host in os.environ.get('PANVISION_INTERNAL_API_HOSTS', '').split(',')
                if host.strip()}
    if parsed.scheme != 'https' and parsed.hostname not in {'localhost', '127.0.0.1', '::1', *internal}:
        raise APIError('The remote API connection must use HTTPS.')
    token = st.session_state.get('api_token') if authenticated else None
    if authenticated and not token:
        raise SessionExpired('Sign in again.')
    try:
        response = httpx.request(method, base + path, headers={'Authorization': f'Bearer {token}'} if token else {},
                                 timeout=timeout, follow_redirects=False, **kwargs)
    except httpx.HTTPError:
        raise APIError('Cannot reach the backend. Check that the FastAPI server is running.') from None
    if response.status_code == 401 and authenticated:
        raise SessionExpired('Your session expired. Sign in again.')
    if response.is_error or response.is_redirect:
        if response.status_code == 413 and path.endswith(('/analysis', '/pancreas', '/lesion')):
            raise APIError('This scan is too large for local model inference.')
        if response.status_code == 503 and path.endswith(('/analysis', '/pancreas', '/lesion')):
            raise APIError('Model inference is unavailable. Check the backend model files and try again.')
        if response.status_code == 422:
            raise APIError('Check the email, study ID, or notes. Notes must be at most 10,000 characters.')
        # Never forward raw proxy errors, tracebacks, or credential-bearing URLs.
        raise APIError({400: 'Check your current password and choose a different password of at least 12 characters.',
                        401: 'Unable to sign in. Check your details or try again in five minutes.',
                        403: 'Administrator access required.', 404: 'Record not found.', 409: 'The operation conflicts with existing data. Check for a duplicate email, unavailable scan, or the last administrator account.',
                        503: 'Database unavailable or not configured.'}.get(response.status_code, 'The backend could not complete this request.'))
    return response


def saved_cases(doctor_id=None):
    # Identity comes from the token; never send a client-supplied doctor ID.
    return request('GET', '/me/cases').json()


def save_case(doctor_id, study_id, notes):
    return request('PUT', f'/cases/{study_id}/notes', json={'notes': notes}).json()


def report_pdf(study_id, notes):
    return request('POST', f'/cases/{study_id}/report.pdf', json={'notes': notes}).content


def sign_out():
    try:
        request('POST', '/auth/logout')
    except SessionExpired:
        pass
    # If backend is unreachable, let the user retry revocation instead of pretending success.
    st.session_state.clear()

