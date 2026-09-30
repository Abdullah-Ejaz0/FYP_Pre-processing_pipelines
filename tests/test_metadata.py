"""Regression tests for metadata.py -- each case here is a real row found in the raw data,
not a hypothetical, so a regression here means real patient data would be mis-parsed again."""

import math

from lucidcxr_prep.metadata import parse_montgomery, parse_shenzhen


def test_montgomery_standard():
    r = parse_montgomery("Patient's Sex: F \nPatient's Age: 027Y\nnormal")
    assert r == {"sex": "F", "age_value": 27, "age_unit": "years", "age_years": 27.0,
                 "diagnosis_text": "normal"}


def test_montgomery_non_mf_sex_passed_through():
    r = parse_montgomery("Patient's Sex: O \nPatient's Age: 005Y\nnormal")
    assert r["sex"] == "O"
    assert r["age_years"] == 5.0


def test_montgomery_same_patient_cross_reference_preserved():
    r = parse_montgomery("Patient's Sex: M \nPatient's Age: 049Y\nNO REPORT (same pt as MCUCXR_0162_1)")
    assert "same pt as MCUCXR_0162_1" in r["diagnosis_text"]


def test_shenzhen_standard():
    r = parse_shenzhen("male 45yrs\nnormal")
    assert r == {"sex": "M", "age_value": 45, "age_unit": "years", "age_years": 45.0,
                 "diagnosis_text": "normal"}


def test_shenzhen_infant_months():
    r = parse_shenzhen("male 16month\nnormal")
    assert r["age_unit"] == "months"
    assert math.isclose(r["age_years"], 16 / 12)


def test_shenzhen_infant_days():
    r = parse_shenzhen("female 64days\nnormal")
    assert r["age_unit"] == "days"
    assert math.isclose(r["age_years"], 64 / 365.25)


def test_shenzhen_no_unit_leaves_age_years_nan():
    r = parse_shenzhen("male 42\nnormal")
    assert r["age_unit"] is None
    assert math.isnan(r["age_years"])


def test_shenzhen_typo_femal():
    r = parse_shenzhen("femal 32yrs\nnormal")
    assert r["sex"] == "F"
    assert r["age_years"] == 32.0


def test_shenzhen_no_space_before_age():
    r = parse_shenzhen("female24yrs\nnormal")
    assert r["sex"] == "F"
    assert r["age_years"] == 24.0


def test_shenzhen_trailing_comma():
    r = parse_shenzhen("male , 29yrs  \t\nbilateral secondary PTB")
    assert r["sex"] == "M"
    assert r["age_years"] == 29.0
