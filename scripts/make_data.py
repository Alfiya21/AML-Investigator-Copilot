import argparse

from aml_copilot.transactions import db, synth


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic accounts and transactions")
    ap.add_argument("--accounts", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    accounts, txns = synth.generate(n_accounts=args.accounts, seed=args.seed)
    db.load_frames(accounts, txns)
    print(f"{len(accounts)} accounts, {len(txns)} transactions, "
          f"{int(accounts.is_suspicious.sum())} suspicious accounts")


if __name__ == "__main__":
    main()
