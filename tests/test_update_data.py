from datetime import date

from scripts import update_data


def _payload(applicable_for: str = "2026-07-07") -> list[dict]:
    return [
        {
            "publicationId": update_data.NTS_D6_PUBLICATION_ID,
            "publicationName": update_data.NTS_D6_NAME,
            "publications": [
                {
                    "applicableAt": "2026-07-13T11:20:00+01:00",
                    "applicableFor": applicable_for,
                    "value": "146.857",
                    "generatedTimeStamp": "2026-07-13T12:00:00+01:00",
                    "qualityIndicator": "",
                }
            ],
        }
    ]


def test_records_from_payload_maps_rest_fields_to_raw_schema():
    records = update_data._records_from_payload(_payload())

    assert records == [
        {
            "Applicable At": "13/07/2026 11:20:00",
            "Applicable For": "07/07/2026",
            "Data Item": "Demand Actual, NTS, D+6",
            "Value": 146.857,
            "Generated Time": "13/07/2026 12:00:00",
            "Quality Indicator": "",
        }
    ]


def test_fetch_demand_records_chunks_long_date_ranges(monkeypatch):
    calls = []

    def fake_fetch(url, payload):
        calls.append(payload)
        return _payload(payload["fromDate"])

    monkeypatch.setattr(update_data, "_fetch_json", fake_fetch)

    result = update_data.fetch_demand_records(
        date(2026, 1, 1),
        date(2026, 3, 15),
    )

    assert len(calls) == 3
    assert len(result) == 3
    assert calls[0]["publicationIds"] == ["PUBOB652"]


def test_records_from_payload_supports_provisional_d1_publication():
    payload = _payload("2026-07-12")
    payload[0]["publicationId"] = update_data.NTS_D1_PUBLICATION_ID
    payload[0]["publicationName"] = update_data.NTS_D1_NAME

    records = update_data._records_from_payload(
        payload,
        publication_id=update_data.NTS_D1_PUBLICATION_ID,
        publication_name=update_data.NTS_D1_NAME,
    )

    assert records[0]["Applicable For"] == "12/07/2026"
    assert records[0]["Data Item"] == "Demand Actual, NTS, D+1"
