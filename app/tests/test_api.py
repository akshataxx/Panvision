import io
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

import numpy as np
import psycopg
import SimpleITK as sitk
from fastapi.testclient import TestClient
from app.api import api


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api)
        self.doctor = {'id': 1, 'email': 'doctor@example.com', 'name': 'Doctor', 'role':'doctor'}
        self.session_patch = patch('app.api.db.session_doctor', side_effect=lambda t, **kwargs: self.doctor if t == 'valid' else None)
        self.session_patch.start()
        self.addCleanup(self.session_patch.stop)
        for target, value in [('record_upload','11111111-1111-4111-8111-111111111111'),('record_report',None)]:
            mocked = patch('app.api.db.'+target, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.headers = {'Authorization':'Bearer valid'}

    def test_authentication_and_secret_redaction(self):
        self.assertEqual(self.client.get('/cases').status_code, 401)
        self.assertEqual(self.client.get('/auth/me', headers={'Authorization':'Bearer expired'}).status_code, 401)
        self.assertEqual(self.client.get('/auth/me', headers=self.headers).json(), self.doctor)
        with patch('app.api.db.authenticate', return_value=self.doctor), patch('app.api.db.new_session', return_value=('opaque-token',12345)):
            response = self.client.post('/auth/login', json={'email':'doctor@example.com','password':'a-secret'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['access_token'],'opaque-token')
        response = self.client.post('/auth/login', json={'email':'bad','password':'DO-NOT-ECHO','unexpected':'SECRET'})
        self.assertEqual(response.status_code,422)
        self.assertNotIn('DO-NOT-ECHO',response.text)
        self.assertNotIn('SECRET',response.text)
        with patch('app.api.db.revoke_session') as revoke:
            self.assertEqual(self.client.post('/auth/logout', headers=self.headers).status_code,204)
            revoke.assert_called_once_with('valid')

    def test_private_notes_and_study_validation(self):
        cases=[{'study_id':'study-1','patient_id':'patient-1','processed_image':'/private/scan.nii.gz'}]
        with patch('app.api.studies',return_value=cases), patch('app.api.db.save_case') as save:
            response=self.client.put('/cases/study-1/notes', headers=self.headers, json={'notes':'Review'})
            self.assertEqual(response.status_code,200)
            save.assert_called_once_with(1,'study-1','Review')
            response=self.client.put('/cases/study-1/notes',headers=self.headers,json={'notes':'x','doctor_id':2})
            self.assertEqual(response.status_code,422)
            self.assertEqual(self.client.put('/cases/study-1/notes',headers=self.headers,json={'notes':'x'*10001}).status_code,422)
            self.assertEqual(self.client.put('/cases/missing/notes',headers=self.headers,json={'notes':'x'}).status_code,404)
            self.assertNotIn('/private',self.client.get('/cases',headers=self.headers).text)
        with patch('app.api.db.saved_cases', return_value={'study-1':'private'}) as read:
            self.assertEqual(self.client.get('/me/cases',headers=self.headers).json(),{'study-1':'private'})
            read.assert_called_once_with(1)
        with patch('app.api.db.session_doctor',side_effect=psycopg.OperationalError('PASSWORD=secret')):
            response=self.client.get('/auth/me',headers=self.headers)
            self.assertEqual(response.status_code,503)
            self.assertNotIn('secret',response.text)

    def test_ct_validation_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'image.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(np.ones((4,8,8),dtype=np.float32)),str(path))
            response=self.client.post('/uploads/validate',headers=self.headers,files={'file':('scan.nii.gz',path.read_bytes())})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['shape_zyx'],[4,8,8])
            response=self.client.post('/uploads/validate',headers=self.headers,files={'file':('scan.nii.gz',b'invalid')})
            self.assertEqual(response.status_code,422)
            response=self.client.post('/uploads/validate',headers=self.headers,files={'file':('scan.mha',b'ElementDataFile = /etc/passwd\n')})
            self.assertEqual(response.status_code,422)
            response=self.client.post('/uploads/report.pdf',headers=self.headers,
                                      files={'file':('scan.nii.gz',path.read_bytes())},data={'notes':'Uploaded scan note'})
            self.assertEqual(response.status_code,200,response.text[:200])
            self.assertTrue(response.content.startswith(b'%PDF'))
            case={'study_id':'study-1','diagnosis':'Non-PDAC','processed_image':str(path)}
            with patch('app.api.studies',return_value=[case]):
                response=self.client.post('/cases/study-1/report.pdf',headers=self.headers,json={'notes':'Research note'})
                self.assertEqual(response.status_code,200)
                self.assertTrue(response.content.startswith(b'%PDF'))
                self.assertEqual(response.headers['content-type'],'application/pdf')

    def test_uploaded_report_accepts_matching_pancreas_mask_only(self):
        uid = '11111111-1111-4111-8111-111111111111'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'scan.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(np.ones((4, 8, 8), dtype=np.float32)), str(path))
            content = path.read_bytes()
            record = {'sha256': hashlib.sha256(content).hexdigest(), 'status': 'validated'}
            mask = np.zeros((4, 8, 8), dtype=bool)
            mask[1, 2:5, 3:6] = True
            with patch('app.api.db.get_upload', return_value=record), \
                 patch('app.api.build_pdf', return_value=b'%PDF-test') as build:
                response = self.client.post('/uploads/report.pdf', headers=self.headers,
                    files={'file': ('scan.nii.gz', content),
                           'pancreas_mask': ('pancreas.mask', np.packbits(mask.reshape(-1)).tobytes())},
                    data={'upload_id': uid})
                self.assertEqual(response.status_code, 200, response.text)
                np.testing.assert_array_equal(build.call_args.kwargs['pancreas_mask'], mask)
                self.assertEqual(self.client.post('/uploads/report.pdf', headers=self.headers,
                    files={'file': ('scan.nii.gz', content), 'pancreas_mask': ('pancreas.mask', b'x')},
                    data={'upload_id': uid}).status_code, 422)

    def test_pancreas_mask_requires_owned_matching_upload(self):
        uid = '11111111-1111-4111-8111-111111111111'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(np.ones((4,8,8), dtype=np.float32)), str(path))
            content = path.read_bytes()
            record = {'status':'validated', 'sha256':hashlib.sha256(content).hexdigest()}
            mask = np.zeros((4,8,8), dtype=bool)
            mask[:,2:5,3:6] = True
            self.assertEqual(self.client.post(f'/uploads/{uid}/pancreas',
                                              files={'file':('scan.nii.gz', content)}).status_code, 401)
            with patch('app.api.db.get_upload', return_value=record) as get_upload, \
                 patch('app.api.segment_pancreas', return_value=mask) as segment:
                response = self.client.post(f'/uploads/{uid}/pancreas', headers=self.headers,
                                            files={'file':('scan.nii.gz', content)})
                self.assertEqual(response.status_code, 200, response.text[:200])
                with np.load(io.BytesIO(response.content), allow_pickle=False) as result:
                    np.testing.assert_array_equal(result['mask'], mask)
                get_upload.assert_called_once_with(UUID(uid), 1)
                segment.assert_called_once()
            with patch('app.api.db.get_upload', side_effect=LookupError), \
                 patch('app.api.segment_pancreas') as segment:
                response = self.client.post(f'/uploads/{uid}/pancreas', headers=self.headers,
                                            files={'file':('scan.nii.gz', content)})
                self.assertEqual(response.status_code, 404)
                segment.assert_not_called()
            with patch('app.api.db.get_upload', return_value={'status':'validated','sha256':'wrong'}), \
                 patch('app.api.segment_pancreas') as segment:
                response = self.client.post(f'/uploads/{uid}/pancreas', headers=self.headers,
                                            files={'file':('scan.nii.gz', content)})
                self.assertEqual(response.status_code, 409)
                segment.assert_not_called()
            from app.pancreas import ModelUnavailable
            with patch('app.api.db.get_upload', return_value=record), \
                 patch('app.api.segment_pancreas', side_effect=ModelUnavailable('secret-path')):
                response = self.client.post(f'/uploads/{uid}/pancreas', headers=self.headers,
                                            files={'file':('scan.nii.gz', content)})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn('secret-path', response.text)

    def test_lesion_route_runs_p3_then_l2_for_owned_upload(self):
        uid = '11111111-1111-4111-8111-111111111111'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(np.ones((4,8,8), dtype=np.float32)), str(path))
            content = path.read_bytes()
            record = {'status':'validated', 'sha256':hashlib.sha256(content).hexdigest()}
            pancreas = np.ones((4,8,8), dtype=bool)
            lesion = np.zeros((4,8,8), dtype=bool)
            lesion[1,2:4,3:5] = True
            with patch('app.api.db.get_upload', return_value=record), \
                 patch('app.api.model_available', return_value=True), \
                 patch('app.api.lesion_model_available', return_value=True), \
                 patch('app.api.segment_pancreas', return_value=pancreas) as p3, \
                 patch('app.api.segment_lesion', return_value=lesion) as l2:
                response = self.client.post(f'/uploads/{uid}/lesion', headers=self.headers,
                                            files={'file':('scan.nii.gz', content)})
                self.assertEqual(response.status_code, 200, response.text[:200])
                with np.load(io.BytesIO(response.content), allow_pickle=False) as result:
                    np.testing.assert_array_equal(result['pancreas'], pancreas)
                    np.testing.assert_array_equal(result['lesion'], lesion)
                p3.assert_called_once()
                l2.assert_called_once()
                np.testing.assert_array_equal(l2.call_args.args[2], pancreas)
            with patch('app.api.db.get_upload', return_value={**record, 'sha256':'wrong'}), \
                 patch('app.api.segment_pancreas') as p3:
                response = self.client.post(f'/uploads/{uid}/lesion', headers=self.headers,
                                            files={'file':('scan.nii.gz', content)})
                self.assertEqual(response.status_code, 409)
                p3.assert_not_called()

    def test_classification_route_runs_p3_then_classifier_for_owned_upload(self):
        from app.classification import NoPancreasRegion
        uid = '11111111-1111-4111-8111-111111111111'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(np.ones((4,8,8), dtype=np.float32)), str(path))
            content = path.read_bytes()
            record = {'status':'validated', 'sha256':hashlib.sha256(content).hexdigest()}
            pancreas = np.ones((4,8,8), dtype=bool)
            result = {'probability': 0.73, 'threshold': 0.42, 'pdac': True, 'slices': 3}
            available = [patch('app.api.model_available', return_value=True),
                         patch('app.api.classifier_model_available', return_value=True)]
            for item in available:
                item.start()
                self.addCleanup(item.stop)
            with patch('app.api.db.get_upload', return_value=record), \
                 patch('app.api.segment_pancreas', return_value=pancreas) as p3, \
                 patch('app.api.classify_pdac', return_value=result) as classifier:
                response = self.client.post(f'/uploads/{uid}/classification', headers=self.headers,
                                            files={'file':('scan.nii.gz', content)})
                self.assertEqual(response.status_code, 200, response.text[:200])
                with np.load(io.BytesIO(response.content), allow_pickle=False) as output:
                    np.testing.assert_array_equal(output['pancreas'], pancreas)
                    self.assertAlmostEqual(float(output['probability']), 0.73)
                    self.assertAlmostEqual(float(output['threshold']), 0.42)
                p3.assert_called_once()
                np.testing.assert_array_equal(classifier.call_args.args[2], pancreas)
            with patch('app.api.db.get_upload', return_value=record), \
                 patch('app.api.segment_pancreas', return_value=pancreas), \
                 patch('app.api.classify_pdac', side_effect=NoPancreasRegion('none')):
                response = self.client.post(f'/uploads/{uid}/classification', headers=self.headers,
                                            files={'file':('scan.nii.gz', content)})
                self.assertEqual(response.status_code, 422)
            with patch('app.api.db.get_upload', return_value={**record, 'sha256':'wrong'}), \
                 patch('app.api.segment_pancreas') as p3:
                response = self.client.post(f'/uploads/{uid}/classification', headers=self.headers,
                                            files={'file':('scan.nii.gz', content)})
                self.assertEqual(response.status_code, 409)
                p3.assert_not_called()

    def test_pdf_classification_uses_server_threshold(self):
        uid = '11111111-1111-4111-8111-111111111111'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(np.ones((4,8,8), dtype=np.float32)), str(path))
            content = path.read_bytes()
            record = {'status':'validated', 'sha256':hashlib.sha256(content).hexdigest()}
            with patch('app.api.db.get_upload', return_value=record), \
                 patch('app.api.frozen_config', return_value={'threshold': 0.42}), \
                 patch('app.api.build_pdf', return_value=b'%PDF-test') as pdf:
                response = self.client.post('/uploads/report.pdf', headers=self.headers,
                                            files={'file':('scan.nii.gz', content)},
                                            data={'upload_id': uid, 'pdac_probability': '0.30'})
                self.assertEqual(response.status_code, 200, response.text[:200])
                self.assertEqual(pdf.call_args.kwargs['classification'], {'probability': 0.30, 'threshold': 0.42})
            response = self.client.post('/uploads/report.pdf', headers=self.headers,
                                        files={'file':('scan.nii.gz', content)},
                                        data={'pdac_probability': '0.30'})
            self.assertEqual(response.status_code, 422)
            response = self.client.post('/uploads/report.pdf', headers=self.headers,
                                        files={'file':('scan.nii.gz', content)},
                                        data={'upload_id': uid, 'pdac_probability': '1.5'})
            self.assertEqual(response.status_code, 422)


    def test_admin_authorization_and_doctor_creation(self):
        for path in ('/admin/accounts','/admin/database'):
            self.assertEqual(self.client.get(path,headers=self.headers).status_code,403)
            self.assertEqual(self.client.get(path).status_code,401)
        with patch('app.api.db.admin_create_doctor') as create:
            response=self.client.post('/admin/doctors',headers=self.headers,
                                      json={'name':'New Doctor','email':'new@example.com','password':'long-password-123'})
            self.assertEqual(response.status_code,403)
            create.assert_not_called()
        self.assertEqual(self.client.post('/auth/register',json={}).status_code,404)
        self.doctor['role']='admin'
        public={'id':2,'name':'New Doctor','email':'new@example.com','role':'doctor','active':1}
        with patch('app.api.db.admin_create_doctor',return_value=public) as create:
            response=self.client.post('/admin/doctors',headers=self.headers,
                                      json={'name':'New Doctor','email':'new@example.com','password':'long-password-123'})
            self.assertEqual(response.status_code,201)
            self.assertNotIn('password',response.text)
            create.assert_called_once_with(1,'new@example.com','New Doctor','long-password-123')
            response=self.client.post('/admin/doctors',headers=self.headers,
                                      json={'name':'New Doctor','email':'new@example.com','password':'short'})
            self.assertEqual(response.status_code,422)
        from app.database import AccountConflict
        with patch('app.api.db.admin_update_account', side_effect=AccountConflict('Last admin protected')):
            response=self.client.patch('/admin/accounts/1',headers=self.headers,json={'active':False,'role':'doctor'})
            self.assertEqual(response.status_code,409)


    def test_password_change_and_history_ownership(self):
        with patch('app.api.db.change_password', return_value=True) as change:
            response=self.client.post('/auth/password',headers=self.headers,
                                      json={'current_password':'old-password-123','new_password':'new-password-123'})
            self.assertEqual(response.status_code,204)
            change.assert_called_once_with(1,'old-password-123','new-password-123')
        with patch('app.api.db.change_password',return_value=False):
            self.assertEqual(self.client.post('/auth/password',headers=self.headers,
                json={'current_password':'wrong','new_password':'new-password-123'}).status_code,400)
        response=self.client.post('/auth/password',headers=self.headers,
                                  json={'current_password':'SECRET-OLD','new_password':'SECRET'})
        self.assertEqual(response.status_code,422)
        self.assertNotIn('SECRET',response.text)
        with patch('app.api.db.upload_history',return_value={'items':[],'total':0}) as history:
            self.assertEqual(self.client.get('/me/uploads?doctor_id=999',headers=self.headers).status_code,200)
            history.assert_called_once_with(1,50,0)
            self.assertEqual(self.client.get('/admin/uploads',headers=self.headers).status_code,403)
        uid='11111111-1111-4111-8111-111111111111'
        with patch('app.api.db.get_upload',side_effect=LookupError) as read:
            self.assertEqual(self.client.get('/uploads/'+uid,headers=self.headers).status_code,404)
            self.assertEqual(read.call_args.args[1],1)
        with patch('app.api.db.save_upload_notes',side_effect=LookupError):
            self.assertEqual(self.client.put('/uploads/'+uid+'/notes',headers=self.headers,json={'notes':'private'}).status_code,404)
        self.doctor['role']='admin'
        with patch('app.api.db.upload_history',return_value={'items':[],'total':0}) as history:
            self.assertEqual(self.client.get('/admin/uploads?doctor_id=2',headers=self.headers).status_code,200)
            history.assert_called_once_with(2,50,0)

    def test_report_rejects_other_upload_or_changed_file(self):
        uid='11111111-1111-4111-8111-111111111111'
        with patch('app.api.db.get_upload',side_effect=LookupError):
            response=self.client.post('/uploads/report.pdf',headers=self.headers,
                                      files={'file':('scan.nii.gz',b'invalid')},data={'upload_id':uid})
            self.assertEqual(response.status_code,404)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'scan.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(np.ones((4,8,8),dtype=np.float32)),str(path))
            with patch('app.api.db.get_upload',return_value={'sha256':'wrong','status':'validated'}):
                response=self.client.post('/uploads/report.pdf',headers=self.headers,
                                          files={'file':('scan.nii.gz',path.read_bytes())},data={'upload_id':uid})
                self.assertEqual(response.status_code,409)


    def test_idle_status_checks_do_not_extend_session(self):
        with patch('app.api.db.session_doctor',return_value=self.doctor) as session:
            self.assertEqual(self.client.get('/auth/session',headers=self.headers).status_code,200)
            session.assert_called_once_with('valid',touch=False)
        with patch('app.api.db.session_doctor',return_value=None):
            self.assertEqual(self.client.get('/auth/session',headers=self.headers).status_code,401)
