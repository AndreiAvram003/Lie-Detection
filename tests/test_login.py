import unittest
from fastapi.testclient import TestClient
from main import app
import uuid

client = TestClient(app)

class TestLogin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.user_data = {
            "username": f"user_{uuid.uuid4().hex[:8]}",
            "email": f"{uuid.uuid4()}@example.com",
            "password": "testpassword"
        }

        register_payload = {
            "username": cls.user_data["username"],
            "email": cls.user_data["email"],
            "password": cls.user_data["password"]
        }

        response = client.post("/auth/register", json=register_payload)
        assert response.status_code == 200

    def test_01_login_success(self):
        login_payload = {
            "email_or_username": self.user_data["email"],
            "password": self.user_data["password"]
        }

        response = client.post("/auth/login", json=login_payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())

    def test_02_login_invalid_password(self):
        login_payload = {
            "email_or_username": self.user_data["email"],
            "password": "wrongpassword"
        }

        response = client.post("/auth/login", json=login_payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())
