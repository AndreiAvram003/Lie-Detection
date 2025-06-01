import unittest
import uuid
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestRegister(unittest.TestCase):

    def test_register_success(self):
        # Generăm un user nou cu date unice
        new_user = {
            "username": "user_" + str(uuid.uuid4())[:8],
            "email": f"{uuid.uuid4()}@example.com",
            "password": "somepassword"
        }

        response = client.post("/auth/register", json=new_user)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("id", data)
        self.assertEqual(data["username"], new_user["username"])
        self.assertEqual(data["email"], new_user["email"])

    def test_register_existing_user(self):
        # Înregistrăm un user o dată
        user = {
            "username": "existinguser",
            "email": "existing@example.com",
            "password": "testpassword"
        }
        client.post("/auth/register", json=user)

        # Încercăm să-l înregistrăm din nou
        response = client.post("/auth/register", json=user)
        self.assertEqual(response.status_code, 400)
        self.assertIn("detail", response.json())
