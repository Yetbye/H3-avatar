import unittest

from aiohttp.test_utils import TestClient, TestServer

from h3_interactive.server import build_app, validate_bind


class ServerCliTest(unittest.IsolatedAsyncioTestCase):
    def test_bind_guard(self):
        validate_bind("127.0.0.1", 8765, False)
        validate_bind("::1", 8765, False)
        validate_bind("localhost", 8765, False)
        with self.assertRaises(ValueError):
            validate_bind("0.0.0.0", 8765, False)
        with self.assertRaises(ValueError):
            validate_bind("127.0.0.1", 0, False)
        validate_bind("0.0.0.0", 8765, True)

    async def test_health_endpoint(self):
        client = TestClient(TestServer(build_app("mock", max_sessions=1)))
        await client.start_server()
        try:
            response = await client.get("/healthz")
            self.assertEqual((response.status, await response.json()), (200, {"status": "ok"}))
        finally:
            await client.close()

    def test_flashhead_requires_runtime_configuration(self):
        with self.assertRaisesRegex(ValueError, "real upstream runtime"):
            build_app("flashhead", max_sessions=1)


if __name__ == "__main__":
    unittest.main()
