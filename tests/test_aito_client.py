from pipeline.aito.client import AitoClient, AitoConfig, api_root, hit_value, unwrap


def test_api_root_master_and_env():
    assert api_root("https://h/db/x/", "") == "https://h/db/x/api/v2"
    assert api_root("https://h/db/x", "v2") == "https://h/db/x/env/v2/api/v2"


def test_hit_value_reads_v2_then_v1():
    assert hit_value({"$value": "great", "feature": "ignored"}) == "great"
    assert hit_value({"feature": "good"}) == "good"


def test_unwrap_scalar_kinds_only():
    assert unwrap({"kind": "evaluation", "data": {"accuracy": 0.3}}) == {"accuracy": 0.3}
    rows = {"offset": 0, "total": 1, "hits": [{}]}
    assert unwrap(rows) is rows


def test_put_schema_creates_collections(monkeypatch):
    sent = []

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    c = AitoClient(AitoConfig("https://h/db/x", "k"))
    monkeypatch.setattr(c._client, "put", lambda path, json: sent.append((path, json)) or R())
    c.put_schema({"schema": {"t": {"type": "table", "columns": {}}, "u": {"columns": {}}}})
    assert sent == [("/schema/t", {"type": "collection", "columns": {}}),
                    ("/schema/u", {"columns": {}, "type": "collection"})]
    c.close()
