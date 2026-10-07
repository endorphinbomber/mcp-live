import pytest

from tests.fake_live import FakeLiveServer
from tonematch.live.client import LiveClient, LiveError
from tonematch.live.session import LiveSession


@pytest.fixture
def live():
    server = FakeLiveServer()
    yield server.live, LiveSession(LiveClient(port=server.port))
    server.close()


def test_plugin_with_is_device_false_is_found(live):
    fake, session = live
    fake.plugins_not_device.add("Kontakt 8")
    assert session.find_browser_item("Kontakt", stock=False) == "query:plugins#Kontakt 8"


def test_falls_back_to_walking_plugins_tree(live):
    fake, session = live
    fake.search_hides.add("Kontakt 8")
    assert session.find_browser_item("Kontakt", stock=False) == "query:plugins#Kontakt 8"
    paths = [i["path"] for i in session.walk_plugins("kontakt")]
    assert paths == ["plugins/VST3/Native Instruments/Kontakt 8"]


def test_explicit_uri_and_helpful_error(live):
    fake, session = live
    assert session.find_browser_item("whatever", stock=False, uri="query:plugins#X") == "query:plugins#X"
    with pytest.raises(LiveError, match="tonematch plugins"):
        session.find_browser_item("Superior Drummer", stock=False)
