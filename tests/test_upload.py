import unittest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestUploadAndHistory(unittest.TestCase):

    def setUp(self):
        self.user_data = {
            "username": "testuploaduser3",
            "email": "testupload3@example.com",
            "password": "testpassword"
        }

        # Înregistrare și login
        client.post("/auth/register", json=self.user_data)
        login_response = client.post("/auth/login", json={
            "email_or_username": self.user_data["email"],
            "password": self.user_data["password"]
        })
        self.token = login_response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_upload_and_history(self):
        video_path = r"C:\Users\Andrei\Desktop\Faculta\Lie Detection\uploaded_videos\1c3f6172-2ab4-4bfb-837a-6e87a81cc88d.mp4"

        # Upload
        with open(video_path, "rb") as video_file:
            upload_response = client.post(
                "/upload/",
                files={"video": ("any_name.mp4", video_file, "video/mp4")},
                headers=self.headers
            )

        self.assertEqual(upload_response.status_code, 200)
        upload_data = upload_response.json()
        self.assertIn("prediction", upload_data)
        self.assertIn("confidence", upload_data)
        self.assertIn("filename", upload_data)

        uploaded_filename = upload_data["filename"]

        # History
        history_response = client.get("/history/", headers=self.headers)
        self.assertEqual(history_response.status_code, 200)

        history_data = history_response.json()
        filenames = [video["url"].split("/")[-1] for video in history_data]

        self.assertIn(uploaded_filename, filenames)
