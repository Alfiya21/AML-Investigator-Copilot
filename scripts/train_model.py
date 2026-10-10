import json

from aml_copilot.transactions import train


def main():
    metrics = train.train()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
