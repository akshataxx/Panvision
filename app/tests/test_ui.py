"""Upload-first UI workflows without database or shared-drive access."""
from pathlib import Path
from types import SimpleNamespace
import io
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1]/'app.py')


class WorkspaceUITests(unittest.TestCase):
    def setUp(self):
        watchdog = patch('app.dashboard.idle_watchdog')
        watchdog.start()
        self.addCleanup(watchdog.stop)

    def test_home_and_report_gate_do_not_scan_dataset(self):
        with patch('app.dashboard.require_doctor', return_value={'id':1,'name':'ABC'}), \
             patch('app.data.case_index') as index:
            app = AppTest.from_file(APP).run()
            self.assertEqual(app.sidebar.radio[0].value, 'Home')
            self.assertFalse(any('DATASET DIAGNOSIS' in m.value for m in app.markdown))
            app.sidebar.radio[0].set_value('Case Report').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('No scan uploaded yet' in m.value for m in app.markdown))
            next(b for b in app.button if b.label == 'Upload CT scan').click().run()
            self.assertEqual(app.sidebar.radio[0].value, 'Upload CT')
            self.assertTrue(next(b for b in app.button if b.label == 'Validate and open report').disabled)
            index.assert_not_called()

    def test_uploaded_report_preview_and_pdf(self):
        with patch('app.dashboard.require_doctor', return_value={'id':1,'name':'ABC'}), \
             patch('app.dashboard.request', return_value=SimpleNamespace(content=b'%PDF-test')) as api:
            app = AppTest.from_file(APP).run()
            app.session_state['upload_review'] = {'digest':'digest', 'name':'scan.nii.gz', 'content':b'ct',
                                                   'array':np.ones((4,8,8)), 'spacing':(1,1,2)}
            app.sidebar.radio[0].set_value('Case Report').run()
            self.assertFalse(app.exception)
            self.assertEqual([tab.label for tab in app.tabs], ['Pancreas', 'Lesion', 'Grad-CAM'])
            self.assertTrue(any('File validation</span><strong>Passed' in item.value and
                                'Model analysis</span><strong>Not run' in item.value
                                for item in app.markdown))
            app.text_area[0].set_value('Review this uploaded scan').run()
            next(b for b in app.button if b.label == 'Prepare PDF').click().run()
            self.assertEqual(app.session_state['uploaded_report_pdf'][1], b'%PDF-test')
            self.assertEqual(api.call_args.args[:2], ('POST','/uploads/report.pdf'))
            self.assertEqual(api.call_args.kwargs['data']['notes'], 'Review this uploaded scan')
            self.assertTrue(any('prediction is unavailable' in info.value for info in app.info))

    def test_review_notes_show_when_changes_are_unsaved(self):
        with patch('app.dashboard.require_doctor', return_value={'id':1,'name':'ABC'}), \
             patch('app.dashboard.request', return_value=SimpleNamespace(content=b'%PDF-test')) as api:
            app = AppTest.from_file(APP).run()
            app.session_state['upload_review'] = {
                'digest':'digest', 'name':'scan.nii.gz', 'content':b'ct',
                'array':np.ones((4,8,8)), 'spacing':(1,1,2), 'upload_id':'upload-1',
            }
            app.sidebar.radio[0].set_value('Case Report').run()
            app.text_area[0].set_value('First note').run()
            self.assertTrue(any('Unsaved changes' in item.value for item in app.warning))

            next(button for button in app.button if button.label == 'Save review notes').click().run()
            api.assert_called_with('PUT', '/uploads/upload-1/notes', json={'notes':'First note'})
            self.assertFalse(any('Unsaved changes' in item.value for item in app.warning))
            self.assertTrue(any('Notes saved' in item.value for item in app.caption))

            app.text_area[0].set_value('Revised note').run()
            self.assertTrue(any('Unsaved changes' in item.value for item in app.warning))
            next(button for button in app.button if button.label == 'Prepare PDF').click().run()
            self.assertFalse(any('Unsaved changes' in item.value for item in app.warning))
            self.assertTrue(any('Notes saved' in item.value for item in app.caption))

    def test_pancreas_segmentation_appears_in_report(self):
        mask = np.zeros((4,8,8), dtype=np.uint8)
        mask[:,2:5,3:6] = 1
        output = io.BytesIO()
        np.savez_compressed(output, mask=mask)
        with patch('app.dashboard.require_doctor', return_value={'id':1,'name':'ABC'}), \
             patch('app.dashboard.request', return_value=SimpleNamespace(content=output.getvalue())) as api:
            app = AppTest.from_file(APP).run()
            app.session_state['upload_review'] = {
                'digest':'digest', 'name':'scan.nii.gz', 'content':b'ct',
                'array':np.ones((4,8,8)), 'spacing':(1,1,2), 'upload_id':'upload-1',
            }
            app.sidebar.radio[0].set_value('Case Report').run()
            next(button for button in app.button if button.label == 'Run pancreas segmentation').click().run()
            self.assertFalse(app.exception, [error.message for error in app.exception])
            api.assert_called_with('POST', '/uploads/upload-1/pancreas',
                                   files={'file':('scan.nii.gz', b'ct')}, timeout=600)
            np.testing.assert_array_equal(app.session_state['upload_review']['pancreas_mask'], mask)
            self.assertTrue(any('Pancreas complete' in item.value for item in app.markdown))

    def test_successful_login_opens_home_without_form(self):
        doctor = {'id':1,'name':'ABC','email':'test@example.com'}
        def response(method, path, **kwargs):
            data = {'access_token':'token','doctor':doctor} if path == '/auth/login' else doctor
            return SimpleNamespace(json=lambda:data)
        with patch('app.auth.request', side_effect=response):
            app=AppTest.from_file(APP).run()
            app.text_input[0].set_value('test@example.com')
            app.text_input[1].set_value('test-password')
            next(b for b in app.button if b.label == 'Sign in').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.sidebar.radio[0].value,'Home')
            self.assertFalse(any(b.label == 'Sign in' for b in app.button))

    def test_validation_button_navigates_to_report(self):
        uploaded = SimpleNamespace(name='scan.nii.gz', getvalue=lambda: b'ct-test')
        def response(method, path, **kwargs):
            if path == '/uploads/validate':
                return SimpleNamespace(json=lambda:{'valid':True, 'upload_id':'upload-1'})
            raise AssertionError(path)
        with patch('app.dashboard.require_doctor', return_value={'id':1,'name':'ABC'}), \
             patch('app.dashboard.request', side_effect=response) as api, \
             patch('app.dashboard.st.file_uploader', return_value=uploaded), \
             patch('app.dashboard.read_ct', return_value=(np.ones((4,8,8)), (1,1,2))):
            app=AppTest.from_file(APP).run()
            app.sidebar.radio[0].set_value('Upload CT').run()
            next(b for b in app.button if b.label == 'Validate and open report').click().run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertEqual(app.sidebar.radio[0].value, 'Case Report')
            self.assertEqual(app.session_state['upload_review']['name'], 'scan.nii.gz')
            self.assertNotIn('pancreas_mask', app.session_state['upload_review'])
            api.assert_called_once_with('POST', '/uploads/validate', files={'file':('scan.nii.gz', b'ct-test')})
            self.assertTrue(any(b.label == 'Run pancreas segmentation' for b in app.button))

    def test_model_tabs_show_only_available_run_action(self):
        with patch('app.dashboard.require_doctor', return_value={'id':1,'name':'ABC'}), \
             patch('app.dashboard.request') as api:
            app=AppTest.from_file(APP).run()
            app.session_state['upload_review'] = {
                'digest':'digest', 'name':'scan.nii.gz', 'content':b'ct',
                'array':np.ones((4,8,8)), 'spacing':(1,1,2), 'upload_id':'upload-1',
            }
            app.sidebar.radio[0].set_value('Case Report').run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertFalse(next(b for b in app.button if b.label == 'Run pancreas segmentation').disabled)
            self.assertTrue(next(b for b in app.button if b.label == 'Run lesion segmentation').disabled)
            self.assertTrue(next(b for b in app.button if b.label == 'Run Grad-CAM').disabled)
            api.assert_not_called()


    def test_doctor_profile_has_no_admin_navigation(self):
        doctor={'id':1,'name':'ABC','email':'abc@example.com','role':'doctor'}
        with patch('app.dashboard.require_doctor',return_value=doctor):
            app=AppTest.from_file(APP).run()
            self.assertNotIn('Administration',app.sidebar.radio[0].options)
            app.sidebar.radio[0].set_value('My Profile').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('abc@example.com' in m.value for m in app.markdown))

    def test_administrator_dashboard(self):
        doctor={'id':1,'name':'Admin','email':'admin@example.com','role':'admin'}
        def response(method,path,**kwargs):
            data=[{**doctor,'active':1}] if path=='/admin/accounts' else {
                'tables':['doctors','doctor_sessions','doctor_cases','admin_audit'],
                'counts':{'accounts':1,'saved_cases':0,'active_sessions':1},'audit':[]}
            if path=='/admin/uploads':
                data={'items':[],'total':0}
            return SimpleNamespace(json=lambda:data)
        with patch('app.dashboard.require_doctor',return_value=doctor),patch('app.dashboard.request',side_effect=response):
            app=AppTest.from_file(APP).run()
            self.assertIn('Administration',app.sidebar.radio[0].options)
            app.sidebar.radio[0].set_value('Administration').run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertTrue(any(b.label=='Create doctor' for b in app.button))


    def test_profile_password_change_clears_sessions(self):
        doctor={'id':1,'name':'ABC','email':'abc@example.com','role':'doctor'}
        with patch('app.dashboard.require_doctor',return_value=doctor),patch('app.dashboard.request') as api:
            app=AppTest.from_file(APP).run()
            app.session_state['api_token']='token'
            app.sidebar.radio[0].set_value('My Profile').run()
            app.text_input[0].set_value('old-password-123')
            app.text_input[1].set_value('new-password-123')
            app.text_input[2].set_value('new-password-123')
            next(b for b in app.button if b.label=='Update password').click().run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            api.assert_called_once_with('POST','/auth/password',json={'current_password':'old-password-123','new_password':'new-password-123'})
            self.assertNotIn('api_token',app.session_state)

    def test_doctor_history_and_activity(self):
        doctor={'id':1,'name':'ABC','email':'abc@example.com','role':'doctor'}
        row={'id':'11111111-1111-4111-8111-111111111111','filename':'scan.nii.gz',
             'doctor_name':'ABC','doctor_email':'abc@example.com','status':'validated',
             'created_at':'2026-10-19T00:00:00Z','reports':1}
        def response(method,path,**kwargs):
            data={'items':[row],'total':1} if path=='/me/uploads' else {
                'events':[{'event':'upload_validated','actor_id':1,'created_at':row['created_at']}], 'notes':'Saved note'}
            return SimpleNamespace(json=lambda:data)
        with patch('app.dashboard.require_doctor',return_value=doctor),patch('app.dashboard.request',side_effect=response):
            app=AppTest.from_file(APP).run()
            app.sidebar.radio[0].set_value('Past Cases').run()
            self.assertFalse(app.exception,[e.message for e in app.exception])
            self.assertEqual(len(app.dataframe),2)
