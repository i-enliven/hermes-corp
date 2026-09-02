"""
Tests for send_image_file() on the base platform adapter,
and MEDIA: .png extraction/routing in the base platform adapter.

Covers: MEDIA: tag extraction for image extensions, and browser screenshot
cleanup throttling (the send_image_file local-file delivery path).

The platform-specific send_image_file tests (Telegram, Discord, Slack) were
removed along with those pruned platform plugins.
"""

import asyncio
import os
import time
from tools.browser_tool import _cleanup_old_screenshots, _last_screenshot_cleanup_by_dir

import pytest

from gateway.platforms.base import BasePlatformAdapter


def _run(coro):
    """Run a coroutine in a fresh event loop for sync-style tests."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# MEDIA: extraction tests for image files
# ---------------------------------------------------------------------------


class TestExtractMediaImages:
    """Test that MEDIA: tags with image extensions are correctly extracted."""

    def test_png_image_extracted(self):
        content = "Here is the screenshot:\nMEDIA:/home/user/.hermes/browser_screenshots/shot.png"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/home/user/.hermes/browser_screenshots/shot.png"
        assert "MEDIA:" not in cleaned
        assert "Here is the screenshot" in cleaned


# ---------------------------------------------------------------------------
# browser_vision screenshot cleanup tests
# ---------------------------------------------------------------------------


class TestScreenshotCleanup:
    def test_cleanup_removes_old_screenshots(self, tmp_path):
        """_cleanup_old_screenshots should remove files older than max_age_hours."""
        _last_screenshot_cleanup_by_dir.clear()

        # Create a "fresh" file
        fresh = tmp_path / "browser_screenshot_fresh.png"
        fresh.write_bytes(b"new")

        # Create an "old" file and backdate its mtime
        old = tmp_path / "browser_screenshot_old.png"
        old.write_bytes(b"old")
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(str(old), (old_time, old_time))

        _cleanup_old_screenshots(tmp_path, max_age_hours=24)

        assert fresh.exists(), "Fresh screenshot should not be removed"
        assert not old.exists(), "Old screenshot should be removed"

    def test_cleanup_is_throttled_per_directory(self, tmp_path):
        _last_screenshot_cleanup_by_dir.clear()

        old = tmp_path / "browser_screenshot_old.png"
        old.write_bytes(b"old")
        old_time = time.time() - (25 * 3600)
        os.utime(str(old), (old_time, old_time))

        _cleanup_old_screenshots(tmp_path, max_age_hours=24)
        assert not old.exists()

        old.write_bytes(b"old-again")
        os.utime(str(old), (old_time, old_time))
        _cleanup_old_screenshots(tmp_path, max_age_hours=24)

        assert old.exists(), "Repeated cleanup should be skipped while throttled"
