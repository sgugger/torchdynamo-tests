import pandas as pd
import os
from collections import defaultdict
import argparse


def generate_and_save_plots(df, output_dir):
    for mixed_precision in set(df["mixed_precision"].values):
        metrics = defaultdict(list)
        filtered_df = df[df["mixed_precision"] == mixed_precision]
        columns = list(df.columns)[2:]

        # saving performance plots
        metrics_columns = [column for column in columns if "time" not in column]
        df_metric = filtered_df[metrics_columns + ["backend"]]
        inductor_backend_values = df_metric[df_metric["backend"] == "inductor"].values[0]
        pytorch_backend_values = df_metric[df_metric["backend"] == "no"].values[0]
        for i, column in enumerate(metrics_columns):
            metrics["metric"].append(column)
            metrics["inductor"].append(inductor_backend_values[i])
            metrics["no"].append(pytorch_backend_values[i])
        df_metric = pd.DataFrame(metrics)
        plot = df_metric.plot.bar(x="metric", rot=0)
        fig = plot.get_figure()
        fig.savefig(os.path.join(output_dir, f"{mixed_precision=}_metric.png"))

        # saving avg time plots
        metrics = defaultdict(list)
        time_columns = [column for column in columns if "time" in column]
        df_time = filtered_df[time_columns + ["backend"]]
        inductor_backend_values = df_time[df_time["backend"] == "inductor"].values[0]
        pytorch_backend_values = df_time[df_time["backend"] == "no"].values[0]
        for i, column in enumerate(time_columns):
            metrics["avg_time"].append(column)
            metrics["inductor"].append(inductor_backend_values[i])
            metrics["no"].append(pytorch_backend_values[i])
        df_metric = pd.DataFrame(metrics)
        plot = df_metric.plot.bar(x="avg_time", rot=0)
        fig = plot.get_figure()
        fig.savefig(os.path.join(output_dir, f"{mixed_precision=}_avg_time.png"))


def get_diff_percentage(df):
    diff_percentage = defaultdict(list)
    for mixed_precision in set(df["mixed_precision"].values):
        diff_percentage["mixed_precision"].append(mixed_precision)
        filtered_df = df[df["mixed_precision"] == mixed_precision]
        columns = list(df.columns)[2:]
        inductor_backend_values = filtered_df[filtered_df["backend"] == "inductor"].values[0][2:]
        pytorch_backend_values = filtered_df[filtered_df["backend"] == "no"].values[0][2:]

        for i, column in enumerate(columns):
            if "time" in column:
                diff_percentage[column].append(
                    str(round((pytorch_backend_values[i] / inductor_backend_values[i]), 2)) + "x"
                )
            else:
                diff_percentage[column].append(
                    str(round((100 * (inductor_backend_values[i] / pytorch_backend_values[i] - 1)), 2)) + "%"
                )
    return pd.DataFrame(diff_percentage)


def main():
    parser = argparse.ArgumentParser(description="Get plots and summary table")
    parser.add_argument("--input_csv_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv_file)
    group_by_columns = ["backend", "mixed_precision"]
    drop_columns = ["num_epochs", "seed"]
    df.drop(columns=drop_columns, inplace=True)
    df = df.groupby(group_by_columns).agg("mean")
    df = df.reset_index()

    generate_and_save_plots(df, args.output_dir)
    diff_df = get_diff_percentage(df)
    file_prefix = args.input_csv_file.split("/")[-1].split(".")[0]
    diff_df.to_csv(os.path.join(args.output_dir, f"{file_prefix}_summary_table.csv"), header=True, index=False)


if __name__ == "__main__":
    main()
