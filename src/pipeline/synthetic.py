"""Deterministic stand-in for the cleaned claims data, used by CI and tests."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TERM_FREQUENCIES = ["monthly", "quarterly", "semi-annual", "annual"]
PROVINCES = ["gauteng", "western cape", "kwazulu-natal", "limpopo", "free state"]
VEHICLE_TYPES = ["passenger vehicle", "medium commercial", "heavy commercial"]
BANKS = ["first national bank", "standard bank", "absa", "nedbank"]


def make_cleaned_data(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Build a frame with the same columns the pipeline reads from cleaned_data.csv."""
    rng = np.random.default_rng(seed)
    cylinders = rng.choice([3, 4, 6, 8], size=n_rows, p=[0.1, 0.6, 0.25, 0.05])
    registration_year = rng.integers(2000, 2016, size=n_rows)
    kilowatts = rng.integers(50, 200, size=n_rows)
    value_estimate = rng.gamma(4.0, 25000.0, size=n_rows).round(2)
    province = rng.choice(PROVINCES, size=n_rows)
    vat_registered = rng.random(n_rows) < 0.2
    alarm = rng.random(n_rows) < 0.5

    risk = (
        0.6 * (cylinders <= 4)
        + 0.5 * vat_registered
        - 0.4 * alarm
        + 0.02 * (2015 - registration_year)
        + 0.3 * (province == "gauteng")
    )
    claim_probability = 1.0 / (1.0 + np.exp(-(risk - 1.6)))
    has_claim = rng.random(n_rows) < claim_probability
    severity = np.exp(rng.normal(8.0 + 1.0 * risk, 0.4))
    total_claims = np.where(has_claim, severity, 0.0).round(2)

    months = pd.to_datetime("2014-02-01") + pd.to_timedelta(
        rng.integers(0, 22, size=n_rows) * 30, unit="D"
    )
    return pd.DataFrame(
        {
            "UnderwrittenCoverID": rng.integers(10000, 300000, size=n_rows),
            "PolicyID": rng.integers(1, 20000, size=n_rows),
            "TransactionMonth": months.strftime("%Y-%m-%d 00:00:00"),
            "IsVATRegistered": vat_registered,
            "Citizenship": np.nan,
            "Gender": rng.choice(["male", "female", "not specified"], size=n_rows),
            "Bank": rng.choice(BANKS, size=n_rows),
            "Province": province,
            "VehicleType": rng.choice(VEHICLE_TYPES, size=n_rows),
            "RegistrationYear": registration_year,
            "Cylinders": cylinders,
            "kilowatts": kilowatts,
            "CustomValueEstimate": value_estimate,
            "AlarmImmobiliser": np.where(alarm, "yes", "no"),
            "Converted": rng.choice(["yes", "no"], size=n_rows, p=[0.05, 0.95]),
            "SumInsured": (value_estimate * rng.uniform(0.8, 1.2, size=n_rows)).round(2),
            "TermFrequency": rng.choice(TERM_FREQUENCIES, size=n_rows),
            "CalculatedPremiumPerTerm": rng.gamma(2.0, 150.0, size=n_rows).round(2),
            "TotalPremium": rng.gamma(2.0, 120.0, size=n_rows).round(2),
            "TotalClaims": total_claims,
            "claim_indicator": has_claim.astype(int),
        }
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Write generated cleaned data to disk.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_cleaned_data(args.rows, args.seed).to_csv(out, index=False)


if __name__ == "__main__":
    main()
