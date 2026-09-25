import unittest

from aiohttp.test_utils import TestClient, TestServer

from h3_interactive import SessionManager
from h3_interactive.backends import MockBackend
from h3_interactive.http_api import create_app


class HttpApiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app = create_app(SessionManager(MockBackend, max_sessions=1))
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def create_session(self, session_id="api-test"):
        response = await self.client.post("/v1/sessions", json={"session_id": session_id})
        self.assertEqual(response.status, 201)
        return (await response.json())["session_id"]

    async def push_audio(self, session_id, sequence, payload=None):
        if payload is None:
            payload = b"\0" * 640
        return await self.client.post(
            f"/v1/sessions/{session_id}/audio",
            data=payload,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Audio-Sequence": str(sequence),
                "X-Audio-PTS-Ms": str(sequence * 20),
            },
        )

    async def test_complete_http_lifecycle(self):
        session_id = await self.create_session()

        response = await self.client.post(
            f"/v1/sessions/{session_id}/control",
            json={"kind": "prompt", "payload": {"text": "wave"}, "revision": 1},
        )
        self.assertEqual(response.status, 204)
        self.assertEqual((await self.push_audio(session_id, 0)).status, 204)
        self.assertEqual((await self.push_audio(session_id, 1)).status, 204)

        response = await self.client.get(f"/v1/sessions/{session_id}/video?timeout=0.1")
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), b"mock:0:0")
        self.assertEqual(response.headers["X-Video-Sequence"], "0")
        self.assertEqual(response.headers["X-Video-PTS-Ms"], "0")
        self.assertEqual(response.headers["X-Control-Revision"], "1")

        response = await self.client.post(f"/v1/sessions/{session_id}/reset")
        self.assertEqual(response.status, 204)
        response = await self.client.get(f"/v1/sessions/{session_id}/status")
        body = await response.json()
        self.assertEqual((response.status, body["state"], body["epoch"]), (200, "active", 1))

        response = await self.client.delete(f"/v1/sessions/{session_id}")
        self.assertEqual(response.status, 204)
        response = await self.client.get(f"/v1/sessions/{session_id}/status")
        self.assertEqual(response.status, 404)

    async def test_validation_timeout_and_capacity_errors(self):
        session_id = await self.create_session()

        response = await self.client.post("/v1/sessions", json={"session_id": "second"})
        self.assertEqual((response.status, (await response.json())["error"]["code"]), (429, "capacity_reached"))

        response = await self.push_audio(session_id, 0, payload=b"bad")
        self.assertEqual((response.status, (await response.json())["error"]["code"]), (400, "invalid_request"))

        response = await self.client.get(f"/v1/sessions/{session_id}/video?timeout=0.001")
        self.assertEqual((response.status, (await response.json())["error"]["code"]), (504, "frame_timeout"))

        response = await self.client.get(f"/v1/sessions/{session_id}/video?timeout=31")
        self.assertEqual(response.status, 400)


if __name__ == "__main__":
    unittest.main()
