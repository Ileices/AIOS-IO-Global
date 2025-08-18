from aios_io.pulsenet import PulseNet


def test_routing_and_status(tmp_path):
    cfg = tmp_path / "peers.json"
    pn = PulseNet(config_path=str(cfg))
    pn.register_peer("peer1", "host", 1, key="secret")
    pn.register_route("ping", "peer1")
    assert pn.routing_table["ping"] == "peer1"
    status = pn.peer_status()
    assert status["peer1"]["has_key"] is True

