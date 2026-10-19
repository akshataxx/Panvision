import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid
import psycopg
from psycopg import sql
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from app.database import create_doctor, authenticate, save_case, saved_cases


@unittest.skipUnless(os.environ.get('PANVISION_TEST_DATABASE_URL'), 'Set PANVISION_TEST_DATABASE_URL for PostgreSQL integration tests')
class DoctorDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.url = os.environ['PANVISION_TEST_DATABASE_URL']
        self.schema = 'test_' + uuid.uuid4().hex
        with psycopg.connect(self.url) as db:
            db.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(self.schema)))
        parts = urlsplit(self.url)
        query = dict(parse_qsl(parts.query))
        query['options'] = '-csearch_path=' + self.schema
        url = urlunsplit(parts._replace(query=urlencode(query)))
        self.env = patch.dict(os.environ, {'PANVISION_DATABASE_URL': url})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        with psycopg.connect(self.url) as db:
            db.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))

    def test_password_login_and_private_notes(self):
        create_doctor('first@example.test', 'First', 'long-test-password-123')
        create_doctor('second@example.test', 'Second', 'another-test-password-123')
        self.assertIsNone(authenticate('first@example.test', 'incorrect'))
        first = authenticate('FIRST@example.test', 'long-test-password-123')
        second = authenticate('second@example.test', 'another-test-password-123')
        save_case(first['id'], 'study-1', 'Private research note')
        self.assertEqual(saved_cases(first['id']), {'study-1': 'Private research note'})
        self.assertEqual(saved_cases(second['id']), {})
        save_case(first['id'], 'study-1', 'Updated')
        self.assertEqual(saved_cases(first['id'])['study-1'], 'Updated')

    def test_account_lockout(self):
        create_doctor('test@example.test', 'Doctor', 'long-test-password-123')
        for _ in range(5):
            self.assertIsNone(authenticate('test@example.test', 'wrong'))
        self.assertIsNone(authenticate('test@example.test', 'long-test-password-123'))

    def test_sqlite_migration_preserves_passwords_and_notes(self):
        import sqlite3
        from app.database import migrate_sqlite, password_hash
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'old.sqlite3'
            with sqlite3.connect(source) as db:
                db.executescript("CREATE TABLE doctors(id,email,name,salt,password_hash,active,failures,locked_until,created_at); "
                                 "CREATE TABLE doctor_cases(doctor_id,study_id,notes,updated_at);")
                salt = bytes(16)
                db.execute('INSERT INTO doctors VALUES (?,?,?,?,?,?,?,?,?)',
                           (7,'old@example.test','Existing',salt,password_hash('existing-password',salt),1,0,0,'2026-01-01 00:00:00'))
                db.execute('INSERT INTO doctor_cases VALUES (?,?,?,?)', (7,'study-old','Preserved','2026-01-01 00:00:00'))
            self.assertEqual(migrate_sqlite(source), (1,1))
            self.assertEqual(authenticate('old@example.test','existing-password')['id'], 7)
            self.assertEqual(saved_cases(7), {'study-old':'Preserved'})
            create_doctor('new@example.test','New','new-password-1234')
            self.assertGreater(authenticate('new@example.test','new-password-1234')['id'],7)
            with self.assertRaises(ValueError):
                migrate_sqlite(source)

    def test_server_sessions_expire_and_revoke(self):
        from app.database import new_session, session_doctor, revoke_session, connect
        import hashlib
        create_doctor('session@example.test', 'Doctor', 'session-password-123')
        doctor = authenticate('session@example.test', 'session-password-123')
        token, expires = new_session(doctor['id'])
        self.assertEqual(session_doctor(token)['id'], doctor['id'])
        with connect() as db:
            stored = db.execute('SELECT token_hash FROM doctor_sessions').fetchone()['token_hash']
        self.assertEqual(stored, hashlib.sha256(token.encode()).hexdigest())
        self.assertNotEqual(stored, token)
        revoke_session(token)
        self.assertIsNone(session_doctor(token))
        token, _ = new_session(doctor['id'])
        with connect() as db:
            db.execute('UPDATE doctor_sessions SET expires_at=0')
        self.assertIsNone(session_doctor(token))

    def test_admin_controls_and_last_admin_protection(self):
        from app.database import admin_create_doctor, admin_update_account, admin_database_status, AccountConflict
        create_doctor('admin@example.test', 'Admin', 'admin-password-123', role='admin')
        admin = authenticate('admin@example.test', 'admin-password-123')
        doctor = admin_create_doctor(admin['id'], 'doctor@example.test', 'Doctor', 'doctor-password-123')
        self.assertEqual(doctor['role'], 'doctor')
        with self.assertRaises(PermissionError):
            admin_create_doctor(doctor['id'], 'blocked@example.test', 'Blocked', 'blocked-password-123')
        with self.assertRaises(AccountConflict):
            admin_update_account(admin['id'], admin['id'], False, 'doctor')
        from app.database import new_session, session_doctor
        token, _ = new_session(doctor['id'])
        admin_update_account(admin['id'], doctor['id'], False, 'doctor')
        self.assertIsNone(session_doctor(token))
        self.assertEqual(len(admin_database_status()['audit']), 2)

    def test_password_change_verifies_old_password_and_revokes_tokens(self):
        from app.database import change_password, new_session, session_doctor
        create_doctor('change@example.test', 'Doctor', 'old-password-123')
        doctor = authenticate('change@example.test', 'old-password-123')
        token, _ = new_session(doctor['id'])
        self.assertFalse(change_password(doctor['id'], 'wrong', 'new-password-123'))
        self.assertIsNotNone(session_doctor(token))
        self.assertTrue(change_password(doctor['id'], 'old-password-123', 'new-password-123'))
        self.assertIsNone(session_doctor(token))
        self.assertIsNone(authenticate('change@example.test', 'old-password-123'))
        self.assertIsNotNone(authenticate('change@example.test', 'new-password-123'))

    def test_upload_history_notes_and_report_are_owner_scoped(self):
        from app.database import record_upload, upload_history, get_upload, save_upload_notes, record_report
        create_doctor('owner@example.test', 'Owner', 'owner-password-123')
        create_doctor('other@example.test', 'Other', 'other-password-123')
        owner = authenticate('owner@example.test', 'owner-password-123')
        other = authenticate('other@example.test', 'other-password-123')
        uid = record_upload(owner['id'], 'scan.nii.gz', 'validated', sha256='digest', size_bytes=42, shape=[4,8,8], spacing=[1.,1.,2.])
        self.assertEqual(upload_history(owner['id'])['total'],1)
        self.assertEqual(upload_history(other['id'])['total'],0)
        self.assertEqual(upload_history()['total'],1)
        with self.assertRaises(LookupError):
            get_upload(uid, other['id'])
        with self.assertRaises(LookupError):
            save_upload_notes(uid, other['id'], 'blocked')
        with self.assertRaises(LookupError):
            record_report(uid, owner['id'], 'different-digest', 'blocked')
        save_upload_notes(uid, owner['id'], 'saved')
        record_report(uid, owner['id'], 'digest', 'report note')
        record = get_upload(uid, owner['id'])
        self.assertEqual(record['notes'], 'report note')
        self.assertEqual([e['event'] for e in record['events']], ['upload_validated','notes_saved','pdf_generated'])
        self.assertEqual(upload_history(owner['id'])['items'][0]['reports'],1)

    def test_twenty_minute_idle_expiry_and_sliding_activity(self):
        from app.database import new_session, session_doctor, SESSION_IDLE_SECONDS
        self.assertEqual(SESSION_IDLE_SECONDS, 1200)
        create_doctor('idle@example.test','Doctor','idle-password-123')
        doctor=authenticate('idle@example.test','idle-password-123')
        with patch('app.database.time.time',return_value=10000):
            token,expires=new_session(doctor['id'])
            self.assertEqual(expires,11200)
        with patch('app.database.time.time',return_value=11199):
            self.assertIsNotNone(session_doctor(token,touch=False))
        with patch('app.database.time.time',return_value=11200):
            self.assertIsNone(session_doctor(token,touch=False))
            self.assertIsNone(session_doctor(token))
        with patch('app.database.time.time',return_value=20000):
            token,_=new_session(doctor['id'])
        with patch('app.database.time.time',return_value=21000):
            self.assertIsNotNone(session_doctor(token))
        with patch('app.database.time.time',return_value=21500):
            self.assertIsNotNone(session_doctor(token,touch=False))
        with patch('app.database.time.time',return_value=22200):
            self.assertIsNone(session_doctor(token))


class DatabaseConfigurationTests(unittest.TestCase):
    def test_requires_postgresql_url(self):
        from app.database import database_url, DatabaseConfigurationError
        for value in ('', 'sqlite:///local.db'):
            with patch.dict(os.environ, {'PANVISION_DATABASE_URL': value}):
                with self.assertRaises(DatabaseConfigurationError):
                    database_url()


class ConnectionDiagnosticTests(unittest.TestCase):
    def test_messages_classify_without_echoing_secrets(self):
        from app.database import connection_error_message
        for raw, expected in [
            ('password authentication failed PASSWORD=TOPSECRET', 'Authentication failed'),
            ('database panvision does not exist PASSWORD=TOPSECRET', 'Database missing'),
            ('connection timeout PASSWORD=TOPSECRET', 'Connection timed out'),
            ('could not translate host PASSWORD=TOPSECRET', 'Hostname lookup failed'),
            ('permission denied PASSWORD=TOPSECRET', 'Permission denied'),
            ('certificate verify failed PASSWORD=TOPSECRET', 'TLS connection failed'),
            ('other PASSWORD=TOPSECRET', 'PostgreSQL connection failed'),
        ]:
            message = connection_error_message(RuntimeError(raw))
            self.assertTrue(message.startswith(expected), message)
            self.assertNotIn('TOPSECRET', message)
