import subprocess


def main():
    # ingest data : Fetch and load data
    subprocess.run(
        [
            "python",
            "src/house_price_prediction/ingest.py",
            "--log-level", "DEBUG",
            "--log-path", "logs/program_logs.txt",
            "--no-console-log",
            "datasets/housing"
        ],
    )

    # train model
    subprocess.run(
        [
            "python",
            "src/house_price_prediction/train.py",
            "--log-level", "DEBUG",
            "--log-path", "logs/program_logs.txt",
            "--no-console-log",
            "datasets/housing"
        ],
    ),

    # # score model
    subprocess.run(
        [
            "python",
            "src/house_price_prediction/score.py",
            "--log-level", "DEBUG",
            "--log-path", "logs/program_logs.txt",
            "--no-console-log",
            "models/final_model.pkl",
            "datasets/housing",

        ],
    )
    return


if __name__ == "__main__":
    main()
