"""SessionSource profile-field roundtrip (multiplex routing residue).

The webhook adapter tests that lived here were purged with the pruned
``gateway.platforms.webhook`` module; only the SessionSource coverage remains.
"""
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, build_session_key


class TestSessionSourceProfileField:
    def test_profile_roundtrips(self):
        s = SessionSource(
            platform=Platform.WEBHOOK if hasattr(Platform, "WEBHOOK") else Platform.TELEGRAM,
            chat_id="c1",
            chat_type="webhook",
            profile="coder",
        )
        restored = SessionSource.from_dict(s.to_dict())
        assert restored.profile == "coder"
