import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout

from src.segmentation import colab_jobs


class ColabJobTests(unittest.TestCase):
    def run_job(self, name, command, logs):
        with redirect_stdout(io.StringIO()) as out:
            ok = colab_jobs.launch(name, command, cwd=logs, logs=logs, poll=0.2)
        return ok, out.getvalue()

    def test_log_is_shown_without_noise_and_result_reported(self):
        with tempfile.TemporaryDirectory() as logs:
            ok, out = self.run_job('t_ok', 'echo "I0000 cpu_feature_guard noise"; echo step 1', logs)
            self.assertTrue(ok)
            self.assertIn('step 1', out)
            self.assertNotIn('cpu_feature_guard', out)
            ok, out = self.run_job('t_fail', 'echo boom; exit 3', logs)
            self.assertFalse(ok)
            self.assertIn('EXIT 3', out)

    def test_running_job_is_not_started_twice_and_can_be_stopped(self):
        with tempfile.TemporaryDirectory() as logs:
            import subprocess
            log = colab_jobs.log_path('t_long', logs)
            subprocess.Popen(['bash', '-c', f'sleep 30; echo done >> {log} # job=t_long'], start_new_session=True)
            time.sleep(0.5)
            self.assertTrue(colab_jobs.running('t_long'))
            self.assertFalse(colab_jobs.running('t_lon'))  # names are matched exactly
            colab_jobs.stop('t_long')
            time.sleep(0.5)
            self.assertFalse(colab_jobs.running('t_long'))


if __name__ == '__main__':
    unittest.main()
