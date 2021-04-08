import plotly
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import argparse
import datetime
import os
import csv
from glob import glob


def plot_evaluations(in_dirs):
    metrics = ['chamfer_p', 'chamfer_n', 'pf_dist']
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=metrics,
                        shared_xaxes='all',
                        vertical_spacing=0.1, horizontal_spacing=0.01,
                        )
    df = dict()
    colors = px.colors.qualitative.Plotly
    for i, exp_dir in enumerate(in_dirs):
        exp_name = os.path.basename(exp_dir.rstrip('/'))
        evals_in_exp = glob(os.path.join(exp_dir, 'vis', 'evaluation*.csv'))
        for eval_f in evals_in_exp:
            eval_name = os.path.splitext(os.path.basename(eval_f))[0]
            with open(eval_f, 'r') as csvfile:
                print(eval_f)
                fieldnames = ['mtime', 'it',
                              'chamfer_p', 'chamfer_n', 'pf_dist']
                for k in fieldnames:
                    df['.'.join([eval_name, exp_name, k])] = []
                reader = csv.DictReader(
                    csvfile, fieldnames=fieldnames, restval='-', )
                for it, row in enumerate(reader):
                    if it == 0 or row['it'] == 0:
                        # skip header
                        continue
                    for k, v in row.items():
                        if v == '-':
                            continue
                        df['.'.join([eval_name, exp_name, k])].append(
                            float(v))

            name_prefix = '.'.join([eval_name, exp_name])
            for idx, k in enumerate(metrics):
                y_data = df[name_prefix + '.' + k]
                x_data = df[name_prefix + '.' + 'mtime']
                fig.add_trace(go.Scatter(x=x_data, y=y_data,
                                         mode='lines+markers',
                                         name=name_prefix + '.' + k,
                                         marker_color=colors[i]),
                              col=1, row=idx + 1)

    fig.update_yaxes(type="log", autorange=True)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ), template='plotly_white')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", type=str, nargs='+', required=True,
                        help="Experiment directories")
    args = parser.parse_args()
    fig = plot_evaluations(args.dirs)
    out_fname = 'eval' + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S") + '.html'
    fig.write_html(
        out_fname)
    print('Saved to ' + out_fname)
