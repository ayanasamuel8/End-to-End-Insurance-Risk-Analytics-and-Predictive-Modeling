"""Stage 1: clean the raw policy data and mark which rows have a claim."""
from pathlib import Path

from src.data_loader import load_data
from src.pipeline.params import load_params, parse_args
from src.preprocessing import clean_data, save_cleaned_data


def clean(params: dict) -> None:
    raw = load_data(params['paths']['raw_data'])
    cleaned = clean_data(raw)
    cleaned['claim_indicator'] = (cleaned['TotalClaims'] > 0).astype(int)
    save_cleaned_data(cleaned, str(Path(params['paths']['cleaned_data'])))


def main(argv=None) -> None:
    clean(load_params(parse_args('Clean the raw data.', argv).params))


if __name__ == '__main__':
    main()
