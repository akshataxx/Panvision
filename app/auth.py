"""Doctor sign-in through the FastAPI backend."""
import streamlit as st
from app.api_client import request, APIError, SessionExpired


@st.fragment(run_every='15s')
def idle_watchdog():
    if not st.session_state.get('api_token'):
        return
    try:
        request('GET','/auth/session')
    except SessionExpired:
        st.session_state.clear()
        st.session_state['login_notice'] = 'You were signed out after inactivity or your session ended. Please sign in again.'
        st.rerun()
    except APIError:
        # Do not leave private scan content visible when the session cannot be verified.
        st.session_state.clear()
        st.session_state['login_notice'] = 'Unable to verify your session. Please sign in again.'
        st.rerun()


def require_doctor():
    # Reuse the same root slot on every rerun so a successful sign-in removes
    # the whole login form before dashboard I/O begins.
    login_slot = st.empty()
    if st.session_state.get('api_token'):
        try:
            doctor = request('GET', '/auth/me').json()
            st.session_state['doctor'] = doctor
            login_slot.empty()
            return doctor
        except SessionExpired:
            st.session_state.clear()
            st.warning('Your session expired. Please sign in again.')
        except APIError as exc:
            st.error(str(exc))
            st.stop()
    notice = st.session_state.pop('login_notice', None)
    if notice:
        st.success(notice)
    ready, setup_message = True, ''
    try:
        request('GET', '/ready', authenticated=False)
    except APIError as exc:
        ready, setup_message = False, str(exc)
    with login_slot.container():
        st.markdown('<div class="hero"><div class="eyebrow">PANVISION</div>'
                    '<h1>Doctor sign in</h1><p class="subtitle">Your research cases, notes, and reports in one workspace.</p></div>',
                    unsafe_allow_html=True)
        _, center, _ = st.columns([1, 2, 1])
        with center, st.container(border=True):
            st.subheader('Welcome back')
            st.write('Sign in to review studies, save private notes, and download reports.')
            if not ready:
                st.warning('Sign-in is waiting for the backend connection.')
                with st.expander('Administrator setup'):
                    st.write(setup_message)
                    st.code('python -m app.database init\npython -m uvicorn app.api:api --host 127.0.0.1 --port 8000', language='bash')
            with st.form('doctor_login', clear_on_submit=True):
                email = st.text_input('Email', max_chars=254)
                password = st.text_input('Password', type='password', max_chars=1024)
                submit = st.form_submit_button('Sign in', type='primary', width='stretch', disabled=not ready)
            if submit:
                try:
                    result = request('POST', '/auth/login', authenticated=False, json={'email': email, 'password': password}).json()
                except APIError as exc:
                    st.error(str(exc))
                else:
                    st.session_state['doctor'] = result['doctor']
                    st.session_state['api_token'] = result['access_token']
                    st.rerun()
            st.caption('Accounts are created by the project administrator. Research prototype only.')

    st.stop()
