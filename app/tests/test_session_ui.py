import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from app.api_client import SessionExpired


class IdleSessionUITests(unittest.TestCase):
    def test_expiry_clears_scan_and_credentials(self):
        script = '''import streamlit as st
from app.auth import idle_watchdog
idle_watchdog()
st.write(st.session_state.get('login_notice','No notice'))
'''
        with patch('app.auth.request',side_effect=SessionExpired('Expired')) as request:
            app=AppTest.from_string(script)
            app.session_state['api_token']='expired-token'
            app.session_state['upload_review']={'name':'private-scan'}
            app.run()
            self.assertFalse(app.exception)
            self.assertNotIn('api_token',app.session_state)
            self.assertNotIn('upload_review',app.session_state)
            self.assertIn('signed out',app.session_state['login_notice'])
            request.assert_called_once_with('GET','/auth/session')
